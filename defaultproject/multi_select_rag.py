"""Conditions A/B/C/D/E for the multi-selection experiment, on one shared base.

    A  mode="all"       the pipeline's five passages, unchanged (= dig_order)
    B  mode="selector"  untrained selector picks a subset
    C  mode="selector"  + adapter_path, the GRPO-trained selector
    D  mode="empty"     always the empty selection
    E  mode="slot5"     always passage 5
    -  mode="oracle"    exactly the label set (uses gold; a reference input, not
                        a bound on achievable answer accuracy)

Retrieval is untouched: `ImprovedRAG.search()` produces the five passages and
their order, and this class only decides which of them reach the reader.

Two calls, one base model. Selection runs with the adapter switched on inside
`selector_active`; the answer generation the harness performs afterwards runs on
the resting state, which is adapter-off. The answer prompt is rebuilt from the
question and the selected passages alone -- the five never survive into it.
"""
import torch

from enhanced_rag import load_dev_answers
from improved_rag import ImprovedRAG
from multi_select_rft import (SLOTS, build_prompt, label_set, parse_sources,
                              passage_flags)
from two_call_reader import (adapter_state, assert_no_baseline_adapter,
                             attach_selector, generate_batch, selector_active)

MODES = ("all", "selector", "empty", "slot5", "oracle")


class MultiSelectRAG(ImprovedRAG):
    def __init__(self, mode="selector", adapter_path=None, dev_path=None,
                 max_new_tokens=24, select_batch_size=8, log_path=None, **kwargs):
        kwargs.setdefault("variant", "v3")
        super().__init__(**kwargs)
        self.log_path = log_path
        if mode not in MODES:
            raise ValueError("mode must be one of %s" % (MODES,))
        self.mode = mode
        self.adapter_path = adapter_path
        # gold is read for the oracle condition and the reporting split only
        self.dev_answers = load_dev_answers(dev_path) if dev_path else {}
        self.max_new_tokens = max_new_tokens
        self.select_batch_size = select_batch_size
        self._selector = None
        self.selection_log = []

    # -- model -------------------------------------------------------------

    def set_model(self, model):
        super().set_model(model)
        # the resting state has to be adapter-off before the harness ever
        # generates, so the adapter is attached at model-set time, not lazily
        if self.mode == "selector" and self.adapter_path:
            assert_no_baseline_adapter(model)
            self._selector = attach_selector(model, self.adapter_path)
            self.model = self._selector          # same weights, adapter resting off

    def adapter_report(self):
        """What the verification script prints; resting state must be off."""
        if self._selector is None:
            return {"adapter": self.adapter_path or "none", "wrapped": False}
        return {"adapter": self.adapter_path, "wrapped": True,
                "resting_enabled": adapter_state(self._selector)}

    # -- call 1: selection ---------------------------------------------------

    def _select(self, questions, contexts):
        prompts = [build_prompt(self.tokenizer, q, c)
                   for q, c in zip(questions, contexts)]
        if self._selector is not None:
            with selector_active(self._selector) as model:
                replies = generate_batch(model, self.tokenizer, prompts,
                                         self.max_new_tokens, self.select_batch_size)
        else:
            # condition B: no adapter exists, so there is nothing to switch on
            replies = generate_batch(self.model, self.tokenizer, prompts,
                                     self.max_new_tokens, self.select_batch_size)
        out = []
        for text in replies:
            sources, valid = parse_sources(text)
            # a malformed reply is not an abstention; it is logged as invalid and
            # falls back to the empty selection, the same rule in B and C
            out.append((sources if valid else [], valid, text))
        return out

    # -- which passages reach the reader -------------------------------------

    def search(self, queries, qids, k=SLOTS):
        passages, scores = super().search(queries, qids, k)
        qids = [str(q) for q in qids]

        if self.mode == "all":
            picks = [(list(range(1, len(c) + 1)), True, "") for c in passages]
        elif self.mode == "empty":
            picks = [([], True, "")] * len(passages)
        elif self.mode == "slot5":
            picks = [([5] if len(c) >= 5 else [], True, "") for c in passages]
        elif self.mode == "oracle":
            picks = []
            for qid, context in zip(qids, passages):
                flags = passage_flags(context, self.dev_answers.get(qid, []))
                picks.append((sorted(label_set(flags)), True, "oracle"))
        else:
            picks = self._select(list(queries), passages)

        out_passages, out_scores = [], []
        for qid, context, score, (sources, valid, raw) in zip(qids, passages, scores, picks):
            flags = passage_flags(context, self.dev_answers.get(qid, []))
            kept = [context[i - 1] for i in sources]
            kept_scores = [score[i - 1] for i in sources]
            self.selection_log.append({
                "qid": qid, "sources": sources, "valid": valid, "raw": raw,
                "n": sum(flags), "flags": flags,
                # the five as actually ranked, and what of them was passed on
                "context_ids": [p.get("id") for p in context],
                "picked_ids": [p.get("id") for p in kept],
                "picked_chars": sum(len(p.get("text", "")) for p in kept),
                # ctx_token_budget is None in this pipeline, so nothing is cut
                "truncated": self.ctx_token_budget is not None,
            })
            out_passages.append(kept)
            out_scores.append(kept_scores)
        return out_passages, out_scores

    @property
    def assembly_stats(self):
        # the harness reads this once at scoring time; persisting here is what
        # makes the per-question selections available for the paired comparison
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
        stats = {"mode": self.mode, "selections": len(log),
                 "valid_format_rate": sum(r["valid"] for r in log) / total,
                 "mean_selected": sum(len(r["sources"]) for r in log) / total}
        stats.update(self.adapter_report())
        if pos:
            stats["n>0_mean_k"] = sum(len(r["sources"]) for r in pos) / len(pos)
            stats["n>0_empty_rate"] = sum(1 for r in pos if not r["sources"]) / len(pos)
        if zero:
            stats["n=0_abstain_rate"] = sum(1 for r in zero if not r["sources"]) / len(zero)
        return stats
