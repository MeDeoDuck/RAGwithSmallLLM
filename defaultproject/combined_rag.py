"""Both axes at once: a trained selector feeding a trained reader.

The two experiments moved different things. `rft_r3` trained the reader and gave
it all five passages; `msel_c` trained a selector and gave the untrained reader
whatever it picked. Neither ran with the other's adapter, so this is the one
combination the results do not cover.

Isolation is by separate model instances rather than named adapters on a shared
base. Two 1.24B copies in bf16 cost about 5 GB on a 16 GB card, which is cheap
next to the risk of a `set_adapter` call in the wrong place: here the selector
simply cannot reach the reader's weights, because they are different objects.

Note this is *not* the two-call resting-off scheme in `two_call_reader.py`. That
module asserts the reader carries no adapter, which is exactly what is false
here -- the reader is the trained one.
"""
import torch

from enhanced_rag import load_dev_answers
from multi_select_rft import SLOTS
from multi_select_rft import build_prompt as selector_prompt
from multi_select_rft import parse_sources, passage_flags
from reader_rft_rag import SourceReaderRAG


class CombinedRAG(SourceReaderRAG):
    """Selector picks a subset; the trained reader answers from it.

    `adapter_path` is the reader's (inherited from SourceReaderRAG, attached to
    self.model in set_model). `selector_adapter` is loaded onto its own copy.
    """

    def __init__(self, selector_adapter=None, dev_path=None, log_path=None,
                 select_max_new_tokens=24, select_batch_size=8, **kwargs):
        super().__init__(**kwargs)
        self.selector_adapter = selector_adapter
        self.log_path = log_path
        # gold is read for the reporting split only; it never enters a prompt
        self.dev_answers = load_dev_answers(dev_path) if dev_path else {}
        self.select_max_new_tokens = select_max_new_tokens
        self.select_batch_size = select_batch_size
        self._selector = None
        self.selection_log = []

    def _load_selector(self):
        if self._selector is not None:
            return self._selector
        import transformers
        base = transformers.AutoModelForCausalLM.from_pretrained(
            self.model.config._name_or_path, torch_dtype=self.model.dtype,
            device_map={"": self.model.device})
        if self.selector_adapter:
            from peft import PeftModel
            base = PeftModel.from_pretrained(base, self.selector_adapter)
        self._selector = base.eval()
        return self._selector

    @torch.no_grad()
    def _select(self, questions, contexts):
        model = self._load_selector()
        prompts = [selector_prompt(self.tokenizer, q, c)
                   for q, c in zip(questions, contexts)]
        picks = []
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            step = max(1, self.select_batch_size)
            for start in range(0, len(prompts), step):
                batch = prompts[start:start + step]
                encoded = self.tokenizer(batch, return_tensors="pt", padding=True,
                                         return_token_type_ids=False).to(model.device)
                out = model.generate(**encoded, do_sample=False,
                                     max_new_tokens=self.select_max_new_tokens,
                                     pad_token_id=self.tokenizer.pad_token_id)
                for reply in out[:, encoded["input_ids"].shape[1]:]:
                    text = self.tokenizer.decode(reply, skip_special_tokens=True)
                    sources, valid = parse_sources(text)
                    # a malformed reply falls back to the empty selection, the
                    # same rule msel_b and msel_c used
                    picks.append((sources if valid else [], valid))
        finally:
            self.tokenizer.padding_side = side
        return picks

    def search(self, queries, qids, k=SLOTS):
        passages, scores = super().search(queries, qids, k)
        qids = [str(q) for q in qids]
        picks = self._select(list(queries), passages)

        out_passages, out_scores = [], []
        for qid, context, score, (sources, valid) in zip(qids, passages, scores, picks):
            flags = passage_flags(context, self.dev_answers.get(qid, []))
            kept = [context[i - 1] for i in sources]
            self.selection_log.append({
                "qid": qid, "sources": sources, "valid": valid,
                "n": sum(flags), "flags": flags,
                "context_ids": [p.get("id") for p in context],
                "picked_ids": [p.get("id") for p in kept],
                "picked_chars": sum(len(p.get("text", "")) for p in kept),
                "truncated": self.ctx_token_budget is not None,
            })
            out_passages.append(kept)
            out_scores.append([score[i - 1] for i in sources])
        return out_passages, out_scores

    @property
    def assembly_stats(self):
        if self.log_path:
            import json as _json
            import os as _os
            _os.makedirs(_os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "w", encoding="utf-8") as handle:
                for record in self.selection_log:
                    handle.write(_json.dumps(record, ensure_ascii=False) + "\n")
        log = self.selection_log
        total = len(log) or 1
        pos = [r for r in log if r["n"] > 0]
        zero = [r for r in log if r["n"] == 0]
        stats = {
            "selector_adapter": self.selector_adapter or "none",
            "reader_adapter": self.adapter_path or "none",
            "selections": len(log),
            "selector_valid_rate": sum(r["valid"] for r in log) / total,
            "mean_selected": sum(len(r["sources"]) for r in log) / total,
        }
        if pos:
            stats["n>0_empty_rate"] = sum(1 for r in pos if not r["sources"]) / len(pos)
        if zero:
            stats["n=0_abstain_rate"] = sum(1 for r in zero if not r["sources"]) / len(zero)
        parent = super().assembly_stats
        if isinstance(parent, dict):
            stats.update({k: v for k, v in parent.items() if k not in stats})
        return stats
