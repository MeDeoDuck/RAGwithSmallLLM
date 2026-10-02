"""Reader conditions A/B/C for the selection experiment, plus two diagnostics.

The reader is the stock v3 reader with its weights untouched. What changes
between conditions is only which passages reach it:

    A  oracle_off=False, selector=None   the existing dig_order top-5 (= dig_order)
    B  selector with no adapter          whatever the untrained selector picks
    C  selector with a trained adapter   whatever the trained selector picks
    -  mode="oracle"                     a perfect selector, using gold (diagnostic)
    -  mode="abstain"                    never any passage (diagnostic)

The two diagnostics bound the whole approach: `oracle` is the best a selector
could ever do here, and `abstain` is what source=0 actually costs. Both read the
gold answer, so neither is system performance.

Isolation: the selector adapter is loaded onto a *second* copy of the base model
rather than toggled on the reader's copy. Toggling is one forgotten context
manager away from silently training-and-reading with the same weights, and the
whole hypothesis is that the reader was not trained.
"""
import torch

from enhanced_rag import load_dev_answers
from improved_rag import ImprovedRAG
from selector_rft import (ABSTAIN, SLOTS, build_prompt, parse_selection,
                          passage_flags)

MODES = ("selector", "oracle", "abstain")


class SelectedPassageRAG(ImprovedRAG):
    """dig_order retrieval, then one passage (or none) is handed to the reader."""

    def __init__(self, mode="selector", adapter_path=None, dev_path=None,
                 max_new_tokens=12, select_batch_size=8, **kwargs):
        kwargs.setdefault("variant", "v3")
        super().__init__(**kwargs)
        if mode not in MODES:
            raise ValueError("mode must be one of %s" % (MODES,))
        self.mode = mode
        self.adapter_path = adapter_path
        # gold is read for the oracle diagnostic and for the reporting split only;
        # it never enters a prompt
        self.dev_answers = load_dev_answers(dev_path) if dev_path else {}
        self.max_new_tokens = max_new_tokens
        self.select_batch_size = select_batch_size
        self._selector = None
        # one row per question, so B and C can be compared pairwise afterwards
        self.selection_log = []

    # -- selector model, kept separate from the reader ------------------------

    def _load_selector(self):
        if self._selector is not None:
            return self._selector
        import transformers
        base = transformers.AutoModelForCausalLM.from_pretrained(
            self.model.config._name_or_path, torch_dtype=self.model.dtype,
            device_map={"": self.model.device})
        if self.adapter_path:
            from peft import PeftModel
            base = PeftModel.from_pretrained(base, self.adapter_path)
        self._selector = base.eval()
        return self._selector

    def reader_is_isolated(self):
        """True when the reader holds no PEFT layers. Called by the check script."""
        return not any(type(m).__name__.startswith("Lora")
                       for m in self.model.modules())

    @torch.no_grad()
    def _choose(self, questions, contexts):
        model = self._load_selector()
        prompts = [build_prompt(self.tokenizer, q, c)
                   for q, c in zip(questions, contexts)]
        picks = []
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        step = max(1, self.select_batch_size)
        for start in range(0, len(prompts), step):
            batch = prompts[start:start + step]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True,
                                     return_token_type_ids=False).to(model.device)
            out = model.generate(**encoded, max_new_tokens=self.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=self.tokenizer.pad_token_id)
            for reply in out[:, encoded["input_ids"].shape[1]:]:
                text = self.tokenizer.decode(reply, skip_special_tokens=True)
                source, valid = parse_selection(text)
                # a malformed reply cannot be read as an abstention: it is logged
                # as invalid and falls back to abstaining, the same rule in B and C
                picks.append((source if valid else ABSTAIN, valid, text))
        self.tokenizer.padding_side = side
        return picks

    # -- the one hook that changes what the reader sees -----------------------

    def search(self, queries, qids, k=SLOTS):
        passages, scores = super().search(queries, qids, k)
        qids = [str(q) for q in qids]
        contexts = passages

        if self.mode == "abstain":
            picks = [(ABSTAIN, True, "")] * len(contexts)
        elif self.mode == "oracle":
            picks = []
            for qid, context in zip(qids, contexts):
                flags = passage_flags(context, self.dev_answers.get(qid, []))
                # a perfect selector under this reward: the first answer-bearing
                # passage when one exists, otherwise abstain
                chosen = next((i + 1 for i, g in enumerate(flags) if g), ABSTAIN)
                picks.append((chosen, True, "oracle"))
        else:
            picks = self._choose(list(queries), contexts)

        out_passages, out_scores = [], []
        for qid, context, score, (source, valid, raw) in zip(qids, contexts, scores, picks):
            flags = passage_flags(context, self.dev_answers.get(qid, []))
            if source == ABSTAIN:
                kept, kept_scores = [], []
            else:
                kept, kept_scores = [context[source - 1]], [score[source - 1]]
            self.selection_log.append({
                "qid": qid, "source": source, "valid": valid, "raw": raw,
                "n": sum(flags), "flags": flags,
                "picked_ids": [p.get("id") for p in kept],
            })
            out_passages.append(kept)
            out_scores.append(kept_scores)
        return out_passages, out_scores

    @property
    def assembly_stats(self):
        log = self.selection_log
        total = len(log) or 1
        pos = [r for r in log if r["n"] > 0]
        zero = [r for r in log if r["n"] == 0]
        stats = {
            "mode": self.mode,
            "adapter": self.adapter_path or "none",
            "selections": len(log),
            "valid_format_rate": sum(r["valid"] for r in log) / total,
            "mean_passages": sum(len(r["picked_ids"]) for r in log) / total,
        }
        if pos:
            stats["n>0_hit"] = sum(
                1 for r in pos if r["source"] != ABSTAIN and r["flags"][r["source"] - 1]) / len(pos)
            stats["n>0_abstained"] = sum(1 for r in pos if r["source"] == ABSTAIN) / len(pos)
        if zero:
            stats["n=0_abstained"] = sum(1 for r in zero if r["source"] == ABSTAIN) / len(zero)
            stats["n=0_picked"] = sum(1 for r in zero if r["source"] != ABSTAIN) / len(zero)
        return stats
