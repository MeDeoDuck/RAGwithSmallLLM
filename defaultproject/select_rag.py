"""A generative selector that replaces the reranker.

UPR and the answer-likelihood proxy both score one passage at a time and sort.
This module instead shows the model the whole BM25 pool and asks it for five
indices in a single pass, which removes the per-passage scoring loop and, unlike
the answer-likelihood proxy, needs no first-round answer at inference time.

The prompt and the parser live here so that training and evaluation cannot drift
apart; `train_selector.py` imports both.
"""
import os
import re

import torch

from enhanced_rag import UPRRerankRAG

SLOTS = 5
MAX_PASSAGE_TOKENS = 100
INSTRUCTION = (
    "Pick the %d passages most likely to contain the answer to the question.\n"
    "Reply with %d numbers separated by commas and nothing else."
) % (SLOTS, SLOTS)


def build_prompt(question, passages, tokenizer, max_passage_tokens=MAX_PASSAGE_TOKENS):
    """The selection prompt: the question, then every candidate with its index."""
    lines = ["Question: %s" % question, ""]
    for index, passage in enumerate(passages, 1):
        text = passage.get("text", "")
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > max_passage_tokens:
            text = tokenizer.decode(ids[:max_passage_tokens], clean_up_tokenization_spaces=False)
        lines.append("[%d] %s. %s" % (index, passage.get("title", ""), text))
    lines += ["", INSTRUCTION, "Selection:"]
    return "\n".join(lines)


def format_target(indices):
    """Zero-based pool indices as the string the model is trained to emit."""
    return " " + ", ".join(str(i + 1) for i in indices)


def parse_selection(text, pool_size, slots=SLOTS):
    """First `slots` distinct in-range indices, zero-based; BM25 order fills any gap.

    A malformed reply must not silently become an empty prompt, so the fallback is
    the plain BM25 top-k -- the same context the no-reranking baseline would use.
    """
    picked = []
    for token in re.findall(r"\d+", text):
        value = int(token) - 1
        if 0 <= value < pool_size and value not in picked:
            picked.append(value)
        if len(picked) == slots:
            break
    valid = len(picked) == slots
    for index in range(pool_size):
        if len(picked) == slots:
            break
        if index not in picked:
            picked.append(index)
    return picked[:slots], valid


class SelectorRAG(UPRRerankRAG):
    """Choose the prompt passages with a LoRA-adapted copy of the generator.

    `adapter_path=None` runs the base model zero-shot, which is the control this
    configuration has to beat before any training is worth reporting.
    """

    def __init__(self, adapter_path=None, k_pool=20, max_new_tokens=24,
                 select_batch_size=4, **kwargs):
        super().__init__(k_pool=k_pool, rerank_enabled=False, **kwargs)
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.select_batch_size = select_batch_size
        self.selector = None
        self.select_stats = {"valid": 0, "total": 0, "reward": 0}

    @property
    def assembly_stats(self):
        total = max(1, self.select_stats["total"])
        return {"adapter": self.adapter_path or "zero-shot",
                "selections": self.select_stats["total"],
                "valid_format_rate": self.select_stats["valid"] / total}

    def _load_selector(self):
        """A second copy of the generator, so the adapter never touches reading."""
        if self.selector is not None:
            return self.selector
        if not self.adapter_path:
            self.selector = self.model
            return self.selector
        import transformers
        from peft import PeftModel
        base = transformers.AutoModelForCausalLM.from_pretrained(
            self.model.name_or_path, torch_dtype=self.model.dtype,
            device_map={"": self.model.device})
        self.selector = PeftModel.from_pretrained(base, self.adapter_path).eval()
        return self.selector

    @torch.no_grad()
    def _select(self, questions, pools):
        model = self._load_selector()
        prompts = [build_prompt(q, p, self.tokenizer) for q, p in zip(questions, pools)]
        chosen = []
        step = max(1, self.select_batch_size)
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        for start in range(0, len(prompts), step):
            batch = prompts[start:start + step]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True,
                                     return_token_type_ids=False).to(model.device)
            out = model.generate(**encoded, max_new_tokens=self.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=self.tokenizer.pad_token_id)
            for row, reply in enumerate(out[:, encoded["input_ids"].shape[1]:]):
                text = self.tokenizer.decode(reply, skip_special_tokens=True)
                picked, valid = parse_selection(text, len(pools[start + row]))
                chosen.append(picked)
                self.select_stats["total"] += 1
                self.select_stats["valid"] += int(valid)
        self.tokenizer.padding_side = side
        return chosen

    def search(self, queries, qids, k=SLOTS):
        pools = self.bm25_pool(queries, [str(q) for q in qids])
        usable = [[p for p in pool if p.get("text")] for pool in pools]
        picks = self._select(list(queries), usable)
        list_passages, list_scores = [], []
        for pool, chosen in zip(usable, picks):
            selected = [pool[i] for i in chosen[:k]]
            list_passages.append([self._strip(p) for p in selected])
            # the selector emits a set, not a ranking; keep BM25 order so slot
            # placement stays a separate, independently measured lever
            list_scores.append([0.0] * len(selected))
        return list_passages, list_scores
