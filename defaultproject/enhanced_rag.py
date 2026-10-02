"""RAG extensions for handout section 4.3.

UPRRerankRAG      deepens the BM25 pool and reranks it with log P(question|passage)
                  computed by the generator itself (Sachan et al., 2022).
AdaptiveAssemblyRAG  reuses those scores to choose how many passages to include,
                  in which order, and whether to drop the context entirely.
SlotOrderRAG      keeps the reranked top-k but permutes it across the prompt slots.
GoldPositionRAG   oracle: forces the answer-bearing passage into one fixed slot.
OracleDIGRerankRAG  oracle: ranks by log P(answer | question, passage) (InfoGain-RAG).
"""

import json
import os

import torch

from prompt_rag import PromptedRAG
from utils.etc import hit2docdict
from utils.metrics import normalize_answer

CLOSED_BOOK_TEMPLATE = "Question: {question}"


class UPRRerankRAG(PromptedRAG):
    """BM25 top-`k_pool` reranked by the generator's query likelihood."""

    scorer_id = "upr"
    # a subclass may point scoring at a different model; reranking only needs an
    # ordering, so the scorer shares neither weights nor vocabulary with the reader
    scorer_model = None
    scorer_tokenizer = None
    scorer_name = None

    def __init__(self, k_pool=20, score_max_passage_tokens=160,
                 passages_per_score_batch=20, cache_path=None, rerank_enabled=True,
                 rerank_cache_path=None, **kwargs):
        super().__init__(**kwargs)
        self.k_pool = k_pool
        self.score_max_passage_tokens = score_max_passage_tokens
        self.passages_per_score_batch = passages_per_score_batch
        self.cache_path = cache_path
        self._cache = None
        self.rerank_enabled = rerank_enabled
        self.rerank_cache_path = rerank_cache_path
        self._rerank_cache = None
        self._rerank_handle = None
        self.rerank_cache_hits = 0
        self._logits_to_keep = True

    # -- BM25 pool -----------------------------------------------------------

    def _load_cache(self):
        if self._cache is not None or not self.cache_path:
            return
        self._cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    passages = []
                    for passage in record["passages"]:
                        # rag_diagnostics writes "docid" and only keeps the text of the first ranks
                        passage.setdefault("id", passage.get("docid"))
                        passage.setdefault("title", "")
                        passage.setdefault("text", "")
                        if passage["text"]:
                            passages.append(passage)
                    self._cache[record["qid"]] = passages

    def bm25_pool(self, queries, qids):
        """Return `k_pool` BM25 passages per query, from cache when available."""
        self._load_cache()
        qids = [str(qid) for qid in qids]
        cached = {}
        missing_q, missing_i = [], []
        for i, qid in enumerate(qids):
            hit = self._cache.get(qid) if self._cache is not None else None
            if hit is not None and len(hit) >= self.k_pool:
                cached[qid] = hit[:self.k_pool]
            else:
                missing_q.append(queries[i])
                missing_i.append(qid)

        if missing_q:
            batch_hits = self.retriever.batch_search(
                queries=list(missing_q), qids=missing_i, k=self.k_pool, threads=8)
            for qid in missing_i:
                passages = []
                for hit in batch_hits[qid]:
                    doc = hit2docdict(hit)
                    title, sep, text = doc["contents"].partition("\n")
                    if not sep:
                        title, text = "", title
                    passages.append({"id": doc.get("id", hit.docid),
                                     "title": title.strip().strip('"'),
                                     "text": text.strip(),
                                     "score": float(hit.score)})
                cached[qid] = passages

        return [cached[qid] for qid in qids]

    # -- rerank-score cache --------------------------------------------------

    @property
    def _sm(self):
        return self.scorer_model if self.scorer_model is not None else self.model

    @property
    def _st(self):
        return self.scorer_tokenizer if self.scorer_tokenizer is not None else self.tokenizer

    def _rerank_signature(self):
        """Scores are only reusable for the same scorer and the same truncation.

        `scorer_id` -- not the class name -- keys the cache, so the subclasses that
        only reorder or reselect (SlotOrderRAG, GoldPositionRAG) share the scores
        UPRRerankRAG paid for, while a different scoring function invalidates them.
        """
        name = self.scorer_name or getattr(self.model, "name_or_path", "?")
        return f"{self.scorer_id}|{name}|{self.score_max_passage_tokens}"

    def _load_rerank_cache(self):
        if self._rerank_cache is not None or not self.rerank_cache_path:
            return
        self._rerank_cache = {}
        if os.path.exists(self.rerank_cache_path):
            signature = self._rerank_signature()
            with open(self.rerank_cache_path, encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if record.get("sig") != signature:
                        continue
                    self._rerank_cache.setdefault(record["qid"], {}).update(record["scores"])

    def _store_rerank_scores(self, qid, docids, scores):
        if not self.rerank_cache_path:
            return
        entry = dict(zip(docids, scores))
        self._rerank_cache.setdefault(qid, {}).update(entry)
        if self._rerank_handle is None:
            os.makedirs(os.path.dirname(self.rerank_cache_path) or ".", exist_ok=True)
            self._rerank_handle = open(self.rerank_cache_path, "a", encoding="utf-8")
        self._rerank_handle.write(json.dumps(
            {"qid": qid, "sig": self._rerank_signature(), "scores": entry}) + "\n")
        self._rerank_handle.flush()

    def _scores_for(self, query, qid, pool):
        """Reranking scores for one pool, served from the cache when complete."""
        self._load_rerank_cache()
        docids = [str(passage.get("id")) for passage in pool]
        if self._rerank_cache is not None:
            hit = self._rerank_cache.get(qid)
            if hit and all(docid in hit for docid in docids):
                self.rerank_cache_hits += 1
                return [hit[docid] for docid in docids]

        scores = []
        step = max(1, self.passages_per_score_batch)
        for start in range(0, len(pool), step):
            scores.extend(self.score_passages(query, pool[start:start + step], qid=qid))
        self._store_rerank_scores(qid, docids, scores)
        return scores

    # -- query-likelihood scoring -------------------------------------------

    def _score_prompt(self, passage):
        text = passage.get("text", "")
        ids = self._st(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.score_max_passage_tokens:
            text = self._st.decode(ids[:self.score_max_passage_tokens],
                                   clean_up_tokenization_spaces=False)
        title = passage.get("title", "")
        title_ids = self._st(title, add_special_tokens=False)["input_ids"]
        if len(title_ids) > 24:
            title = self._st.decode(title_ids[:24], clean_up_tokenization_spaces=False)
        return (f"Passage: {title}. {text}\n"
                "Please write a question based on this passage.\n"
                "Question:")

    @torch.no_grad()
    def _mean_logprob(self, prefixes, target_text, prefix_len):
        """Mean log P(target token | prefix), one score per prefix, teacher-forced.

        Dividing by the target length does not change the ranking inside one query
        (the scored tokens are identical) but puts every query on the same scale,
        which AdaptiveAssemblyRAG's thresholds depend on.
        """
        device = self._sm.device
        target_ids = self._st(target_text, add_special_tokens=False)["input_ids"]
        if not target_ids or not prefixes:
            return [0.0] * len(prefixes)
        t_len = len(target_ids)
        t_tensor = torch.tensor(target_ids, device=device).unsqueeze(0)

        self._st.padding_side = "left"
        prefix = self._st(
            prefixes, padding="max_length", max_length=prefix_len, truncation=True,
            return_tensors="pt", return_token_type_ids=False,
        )
        batch = prefix["input_ids"].size(0)
        input_ids = torch.cat(
            [prefix["input_ids"].to(device), t_tensor.expand(batch, t_len)], dim=1)
        attention_mask = torch.cat(
            [prefix["attention_mask"].to(device),
             torch.ones((batch, t_len), dtype=prefix["attention_mask"].dtype, device=device)],
            dim=1)

        outputs = None
        if self._logits_to_keep:
            try:
                outputs = self._sm(input_ids=input_ids, attention_mask=attention_mask,
                                     logits_to_keep=t_len + 1)
            except TypeError:
                self._logits_to_keep = False
        if outputs is None:
            outputs = self._sm(input_ids=input_ids, attention_mask=attention_mask)

        # GPT-small returns (logits, past_key_values); HF models return a dataclass
        raw = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        # slice before float(): the full [B, L, V] tensor must not be upcast
        logits = raw[:, -(t_len + 1):-1, :]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        target = input_ids[:, -t_len:].unsqueeze(-1)
        scores = logprobs.gather(-1, target).squeeze(-1).sum(-1) / t_len
        return scores.tolist()

    def score_passages(self, query, passages, qid=None):
        """Mean log P(query | passage) -- the UPR score of Sachan et al. (2022)."""
        return self._mean_logprob([self._score_prompt(p) for p in passages],
                                  " " + query, self.score_max_passage_tokens + 80)

    # -- ModelRAG interface --------------------------------------------------

    @staticmethod
    def _strip(passage):
        return {"id": passage.get("id"), "title": passage.get("title", ""),
                "text": passage.get("text", "")}

    def ranked_pool(self, queries, qids):
        """Per query, the whole pool in final rank order plus its scores.

        Subclasses that need the passages below the top-k -- gold-position
        injection needs clean distractors -- read the pool from here instead of
        re-running the search.
        """
        pools = self.bm25_pool(queries, qids)
        out = []
        for query, qid, pool in zip(queries, [str(q) for q in qids], pools):
            if self.rerank_enabled and pool:
                scores = self._scores_for(query, qid, pool)
                order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
            else:
                scores = [p["score"] for p in pool]
                order = list(range(len(pool)))
            out.append(([pool[i] for i in order], [scores[i] for i in order]))
        return out

    def search(self, queries, qids, k=5):
        list_passages, list_scores = [], []
        for passages, scores in self.ranked_pool(queries, qids):
            list_passages.append([self._strip(p) for p in passages[:k]])
            list_scores.append(scores[:k])
        return list_passages, list_scores


SLOT_ORDERS = ("desc", "asc", "edge", "edge_last")


def slot_permutation(name, n):
    """Rank indices in prompt-slot order; rank 0 is the best-scored passage.

    "edge" and "edge_last" deal the ranks alternately to the two ends, so the
    strongest passages land on the first and last slot and the weakest in the
    middle -- the arrangement the position measurement on this dev set favours.
    """
    ranks = list(range(n))
    if name == "desc":
        return ranks
    if name == "asc":
        return ranks[::-1]
    if name not in ("edge", "edge_last"):
        raise ValueError(f"unknown slot order {name!r}; choose from {SLOT_ORDERS}")
    front, back = [], []
    best_last = name == "edge_last"
    for i, rank in enumerate(ranks):
        (back if (i % 2 == 0) == best_last else front).append(rank)
    return front + back[::-1]


def load_dev_answers(dev_path):
    """{uid: answers} using RAGDataset's own uid rule, so it joins with the outputs."""
    from datasets import Dataset as HF_Dataset

    dataset = HF_Dataset.load_from_disk(dev_path)
    if "uid" in dataset.column_names:
        uids = [str(uid) for uid in dataset["uid"]]
    else:
        uids = [f"eval_{i:04d}" for i in range(len(dataset))]
    return dict(zip(uids, dataset["answers"]))


def passage_has_answer(passage, answers_norm):
    text = normalize_answer(f"{passage.get('title', '')} {passage.get('text', '')}")
    return any(answer and answer in text for answer in answers_norm)


class SlotOrderRAG(UPRRerankRAG):
    """The reranked top-k, permuted across the prompt slots.

    Nothing is added or dropped, so retrieval quality is held exactly fixed and
    only the position of each passage in the prompt changes.
    """

    def __init__(self, slot_order="edge", **kwargs):
        super().__init__(**kwargs)
        if slot_order not in SLOT_ORDERS:
            raise ValueError(f"unknown slot order {slot_order!r}; choose from {SLOT_ORDERS}")
        self.slot_order = slot_order

    def search(self, queries, qids, k=5):
        list_passages, list_scores = super().search(queries, qids, k=k)
        out_passages, out_scores = [], []
        for passages, scores in zip(list_passages, list_scores):
            order = slot_permutation(self.slot_order, len(passages))
            out_passages.append([passages[i] for i in order])
            out_scores.append([scores[i] for i in order])
        return out_passages, out_scores


class GoldPositionRAG(UPRRerankRAG):
    """Oracle control: the answer-bearing passage is forced into `gold_slot`.

    The other slots are filled with the highest-ranked passages that do NOT
    contain the answer, so a sweep over gold_slot varies position and nothing
    else. Questions whose pool has no answer-bearing passage, or too few clean
    distractors, keep the plain top-k and are counted in `skipped`; compare the
    slots on the injected subset only.
    """

    def __init__(self, gold_slot=0, dev_path=None, **kwargs):
        super().__init__(**kwargs)
        self.gold_slot = gold_slot
        self.answers = load_dev_answers(dev_path) if dev_path else {}
        self.injected = 0
        self.skipped = 0

    @property
    def assembly_stats(self):
        total = self.injected + self.skipped
        return {"gold_slot": self.gold_slot, "injected": self.injected,
                "skipped": self.skipped,
                "inject_rate": self.injected / total if total else 0.0}

    def reset_stats(self):
        self.injected = 0
        self.skipped = 0

    def search(self, queries, qids, k=5):
        list_passages, list_scores = [], []
        for (passages, scores), qid in zip(self.ranked_pool(queries, qids),
                                           [str(q) for q in qids]):
            answers_norm = [normalize_answer(a) for a in self.answers.get(qid, [])]
            flags = [passage_has_answer(p, answers_norm) for p in passages]
            gold = next((i for i, flag in enumerate(flags) if flag), None)
            clean = [i for i, flag in enumerate(flags) if not flag]

            if gold is None or len(clean) < k - 1 or not 0 <= self.gold_slot < k:
                self.skipped += 1
                chosen = list(range(min(k, len(passages))))
            else:
                self.injected += 1
                fillers = clean[:k - 1]
                chosen = fillers[:self.gold_slot] + [gold] + fillers[self.gold_slot:]

            list_passages.append([self._strip(passages[i]) for i in chosen])
            list_scores.append([scores[i] for i in chosen])
        return list_passages, list_scores


class OracleDIGRerankRAG(UPRRerankRAG):
    """Oracle ceiling for InfoGain-RAG: rank by log P(gold answer | question, passage).

    DIG subtracts a no-document baseline, but that term is constant within one
    query, so it cannot change the ranking; it is recorded only so the sign of
    DIG can be reported. This uses the gold answer and is therefore a diagnostic
    upper bound, never a method: no system may be scored with it.
    """

    scorer_id = "oracle_dig"

    def __init__(self, dev_path=None, dig_dump_path=None, **kwargs):
        super().__init__(**kwargs)
        self.answers = load_dev_answers(dev_path) if dev_path else {}
        self.dig_dump_path = dig_dump_path
        self._dump_handle = None

    def _target(self, qid):
        answers = self.answers.get(str(qid), [])
        # only the first alias is scored, which keeps the cost equal to UPR and
        # makes the ceiling conservative rather than optimistic
        return " " + (answers[0].strip() if answers else "")

    def _dig_prompt(self, query, passage):
        text = passage.get("text", "")
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.score_max_passage_tokens:
            text = self.tokenizer.decode(ids[:self.score_max_passage_tokens],
                                         clean_up_tokenization_spaces=False)
        return (f"Title: {passage.get('title', '')}\nPassage: {text}\n"
                f"Question: {query}\nAnswer:")

    def score_passages(self, query, passages, qid=None):
        return self._mean_logprob([self._dig_prompt(query, p) for p in passages],
                                  self._target(qid), self.score_max_passage_tokens + 96)

    def ranked_pool(self, queries, qids):
        out = super().ranked_pool(queries, qids)
        if not self.dig_dump_path:
            return out
        if self._dump_handle is None:
            os.makedirs(os.path.dirname(self.dig_dump_path) or ".", exist_ok=True)
            self._dump_handle = open(self.dig_dump_path, "a", encoding="utf-8")
        for (passages, scores), query, qid in zip(out, queries, [str(q) for q in qids]):
            baseline = self._mean_logprob(
                [f"Question: {query}\nAnswer:"], self._target(qid),
                self.score_max_passage_tokens + 96)[0]
            self._dump_handle.write(json.dumps({
                "qid": qid, "baseline": baseline,
                "docids": [str(p.get("id")) for p in passages],
                "dig": scores,
            }) + "\n")
        self._dump_handle.flush()
        return out


class AdaptiveAssemblyRAG(UPRRerankRAG):
    """Score-driven choice of how many passages to include, in which order."""

    def __init__(self, order="asc", tau_mass=0.8, k_min=1, k_max=5,
                 tau_low=None, temperature="auto", **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.tau_mass = tau_mass
        self.k_min = k_min
        self.k_max = k_max
        self.tau_low = tau_low
        self.score_temperature = temperature
        self._gated = 0
        self._selected_counts = []

    def score_probs(self, scores):
        """Softmax over the scores, made scale-free so one tau works for BM25 and UPR.

        BM25 scores live around +20 while mean log-likelihoods live around -2.5;
        dividing by the spread keeps tau_mass and tau_low comparable across both.
        """
        tensor = torch.tensor(scores, dtype=torch.float32)
        if self.score_temperature == "auto":
            temperature = max(tensor.std().item(), 1e-3) if len(scores) > 1 else 1.0
        else:
            temperature = self.score_temperature
        return torch.softmax((tensor - tensor.max()) / temperature, dim=0)

    def assemble(self, scores):
        """Return (selected indices in prompt order, gate_fired)."""
        if not scores:
            return [], False
        if self.k_min > self.k_max:
            raise ValueError(f"k_min ({self.k_min}) must not exceed k_max ({self.k_max})")
        probs = self.score_probs(scores)
        if self.tau_low is not None and probs.max().item() < self.tau_low:
            return [], True

        ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        total, selected = 0.0, []
        for idx in ranking:
            selected.append(idx)
            total += probs[idx].item()
            if len(selected) >= self.k_max:
                break
            if len(selected) >= self.k_min and total >= self.tau_mass:
                break
        if self.order == "asc":
            selected = list(reversed(selected))
        return selected, False

    @property
    def assembly_stats(self):
        n = len(self._selected_counts)
        return {
            "n": n,
            "gate_fired": self._gated,
            "gate_rate": self._gated / n if n else 0.0,
            "avg_passages": sum(self._selected_counts) / n if n else 0.0,
        }

    def reset_stats(self):
        self._gated = 0
        self._selected_counts = []

    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        list_passages, list_scores = self.search(queries, qids, k=max(k, self.k_max))
        texts = []
        self.last_retrieved = list_passages
        self.last_contexts = []
        for query, passages, scores in zip(queries, list_passages, list_scores):
            selected, gate = self.assemble(scores)
            self._gated += int(gate)
            self._selected_counts.append(len(selected))
            chosen = [passages[i] for i in selected]
            self.last_contexts.append(chosen)
            if not chosen:
                user_text = CLOSED_BOOK_TEMPLATE.format(question=query)
            else:
                user_text = None
            if not self.use_chat_template:
                if chosen:
                    texts.append(f"{self.format_passages(chosen)}\nQuestion: {query}\nAnswer:")
                else:
                    texts.append(f"Question: {query}\nAnswer:")
                continue
            if user_text is None:
                messages = self.build_messages(query, chosen)
            else:
                messages = self.build_messages(query, [])
                messages[-1]["content"] = user_text
            texts.append(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                date_string=self.date_string))
        return texts
