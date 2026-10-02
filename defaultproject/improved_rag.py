"""Three published ideas applied to the k_pool problem (handout section 4.3).

Deepening the BM25 pool raises hasanswer but converts less and less of it into
accuracy while the cost grows linearly (k_pool 50 -> 100 buys +0.70pp for 66
extra minutes, and is not significant). Each paper attacks a different part of
that:

EXIT (Findings-ACL 2025)   the token budget cannot hold a deep pool
                           -> keep sentences, not passages, so a deep pool fits
                              in the token budget of a shallow one
InfoGain-RAG (EMNLP 2025)  query-likelihood ranks relevance, not answer-bearing
                           -> also score how much a passage raises the likelihood
                              of an answer, and fuse the two rankings
GRAD (ACL 2026)            the reader cannot use what the deeper pool retrieves
                           -> reorder the prompt against the measured position bias

Neither paper's trained component is used: EXIT's sentence classifier, InfoGain's
reranker and GRAD's debiased SLMs all require training this project has no budget
for, and none of the three released usable weights. What is borrowed is the idea.
"""

import json
import os

import torch

from enhanced_rag import UPRRerankRAG, slot_permutation

SENTENCE_PROMPT = """Query:
{query}
Full context:
Title: {title}
Passage: {passage}
Sentence:
{sentence}
Is this sentence useful in answering the query? Answer only "Yes" or "No"."""

SENTENCE_SYSTEM = ("You judge whether a single sentence helps answer a question. "
                   'Answer with exactly one word, "Yes" or "No".')


def split_sentences(text):
    from nltk.tokenize import sent_tokenize

    return [s.strip() for s in sent_tokenize(text) if s.strip()]


class ScoreCache:
    """Append-only {key: score} cache on disk, one jsonl line per query."""

    def __init__(self, path, signature):
        self.path = path
        self.signature = signature
        self._entries = None
        self._handle = None
        self.hits = 0

    def _load(self):
        if self._entries is not None:
            return
        self._entries = {}
        if self.path and os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue  # a torn line from an interrupted run must not be fatal
                    if record.get("sig") == self.signature:
                        self._entries.setdefault(record["qid"], {}).update(record["scores"])

    def get(self, qid, keys):
        self._load()
        hit = self._entries.get(qid)
        if hit and all(key in hit for key in keys):
            self.hits += 1
            return [hit[key] for key in keys]
        return None

    def put(self, qid, keys, scores):
        if not self.path:
            return
        self._load()
        entry = dict(zip(keys, scores))
        self._entries.setdefault(qid, {}).update(entry)
        if self._handle is None:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._handle = open(self.path, "a", encoding="utf-8")
        self._handle.write(json.dumps({"qid": qid, "sig": self.signature,
                                       "scores": entry}) + "\n")
        self._handle.flush()


class ImprovedRAG(UPRRerankRAG):
    """UPR reranking plus any combination of the three borrowed ideas.

    compress_from  EXIT: pick sentences out of this many reranked passages
    use_proxy      InfoGain: fuse UPR with a first-pass answer likelihood
    slot_order     GRAD: where the best evidence sits in the prompt
    """

    def __init__(self, compress_from=None, token_budget=731, max_blocks=None,
                 sentence_cache_path=None, sentences_per_score_batch=32,
                 passages_per_score_batch=50, max_sentence_tokens=96,
                 use_proxy=False, pseudo_path=None, proxy_cache_path=None, rrf_k=60, fuse=True,
                 slot_order="desc", **kwargs):
        super().__init__(passages_per_score_batch=passages_per_score_batch, **kwargs)
        self.compress_from = compress_from
        self.token_budget = token_budget
        self.max_blocks = max_blocks
        # batch size changes the scores themselves (bf16, two-logit softmax), so it
        # is fixed here and carried in the cache signature rather than left tunable
        self.sentences_per_score_batch = sentences_per_score_batch
        self.max_sentence_tokens = max_sentence_tokens
        self.use_proxy = use_proxy
        self.rrf_k = rrf_k
        self.fuse = fuse
        self.slot_order = slot_order
        self._sentence_cache_path = sentence_cache_path
        self._proxy_cache_path = proxy_cache_path
        self._sentence_cache = None
        self._proxy_cache = None
        self._yes_no = None
        self.pseudo = self._load_pseudo(pseudo_path) if use_proxy else {}
        self._blocks = []
        self._tokens = []
        self._sentences = []

    # -- first-pass answers ---------------------------------------------------

    @staticmethod
    def _load_pseudo(path):
        """{qid: first-pass answer} from a previous run's output file."""
        if not path or not os.path.exists(path):
            raise ValueError(f"use_proxy needs a first-pass output file; {path!r} not found")
        pseudo = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 4:
                    pseudo[fields[0]] = fields[3].strip()
        return pseudo

    # -- separate scorer ------------------------------------------------------

    def set_scorer(self, model, tokenizer, name):
        """Score passages with a different model from the one that generates.

        Reranking only has to produce an ordering, so the scorer shares neither
        weights nor vocabulary with the reader; nothing is added in logit space.
        A 63M scorer costs about a twentieth of the 1.2B one per passage.
        """
        self.scorer_model = model
        self.scorer_tokenizer = tokenizer
        self.scorer_name = name
        return self

    # -- caches ---------------------------------------------------------------

    def _score_cache(self, attr, path, kind, batch):
        """Cache handle for one scorer.

        The batch size belongs in the signature. P(Yes) is a two-way softmax of
        two bf16 logits, so a different batch shape -- a different kernel and a
        different accumulation order -- moves it by up to 4e-2, which was
        measured to change 12% of the selected sentences. Scores from two batch
        sizes must therefore never be mixed inside one comparison.
        """
        cache = getattr(self, attr)
        if cache is None:
            signature = (f"{kind}|{getattr(self.model, 'name_or_path', '?')}"
                         f"|{self.score_max_passage_tokens}|b{batch}")
            cache = ScoreCache(path, signature)
            setattr(self, attr, cache)
        return cache

    # -- InfoGain: answer likelihood ------------------------------------------

    def _answer_prompt(self, query, passage):
        text = passage.get("text", "")
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.score_max_passage_tokens:
            text = self.tokenizer.decode(ids[:self.score_max_passage_tokens],
                                         clean_up_tokenization_spaces=False)
        return (f"Title: {passage.get('title', '')}\nPassage: {text}\n"
                f"Question: {query}\nAnswer:")

    def _proxy_scores(self, query, qid, pool):
        """mean log P(first-pass answer | question, passage), cached per passage."""
        cache = self._score_cache("_proxy_cache", self._proxy_cache_path, "proxy_dig",
                                  self.passages_per_score_batch)
        keys = [str(p.get("id")) for p in pool]
        hit = cache.get(qid, keys)
        if hit is not None:
            return hit
        answer = self.pseudo.get(qid, "").strip()
        if not answer:
            # no first-pass answer: fall back to a flat score so RRF sees only UPR
            return [0.0] * len(pool)
        scores = []
        step = max(1, self.passages_per_score_batch)
        for start in range(0, len(pool), step):
            chunk = pool[start:start + step]
            scores.extend(self._mean_logprob(
                [self._answer_prompt(query, p) for p in chunk],
                " " + answer, self.score_max_passage_tokens + 96))
        cache.put(qid, keys, scores)
        return scores

    @staticmethod
    def _rrf(rankings, k):
        """Reciprocal rank fusion; robust when one of the rankers is noisy."""
        n = len(rankings[0])
        fused = [0.0] * n
        for scores in rankings:
            order = sorted(range(n), key=lambda i: scores[i], reverse=True)
            for rank, index in enumerate(order):
                fused[index] += 1.0 / (k + rank + 1)
        return fused

    def ranked_pool(self, queries, qids):
        if not self.use_proxy:
            return super().ranked_pool(queries, qids)
        pools = self.bm25_pool(queries, qids)
        out = []
        for query, qid, pool in zip(queries, [str(q) for q in qids], pools):
            if not pool:
                out.append(([], []))
                continue
            upr = self._scores_for(query, qid, pool)
            proxy = self._proxy_scores(query, qid, pool)
            # fuse=False isolates the answer-likelihood signal from the fusion
            fused = self._rrf([upr, proxy], self.rrf_k) if self.fuse else proxy
            order = sorted(range(len(pool)), key=lambda i: fused[i], reverse=True)
            out.append(([pool[i] for i in order], [fused[i] for i in order]))
        return out

    # -- EXIT: sentence selection ---------------------------------------------

    def _yes_no_ids(self):
        if self._yes_no is None:
            ids = []
            for word in ("Yes", "No"):
                encoded = self.tokenizer(word, add_special_tokens=False)["input_ids"]
                if len(encoded) != 1:
                    raise ValueError(f"{word!r} is not a single token for this tokenizer; "
                                     "the two-way softmax would score the wrong position")
                ids.append(encoded[0])
            self._yes_no = tuple(ids)
        return self._yes_no

    def _sentence_prompt(self, query, passage, sentence):
        text = passage.get("text", "")
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.score_max_passage_tokens:
            text = self.tokenizer.decode(ids[:self.score_max_passage_tokens],
                                         clean_up_tokenization_spaces=False)
        sentence_ids = self.tokenizer(sentence, add_special_tokens=False)["input_ids"]
        if len(sentence_ids) > self.max_sentence_tokens:
            sentence = self.tokenizer.decode(sentence_ids[:self.max_sentence_tokens],
                                             clean_up_tokenization_spaces=False)
        user = SENTENCE_PROMPT.format(query=query, title=passage.get("title", ""),
                                      passage=text, sentence=sentence)
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SENTENCE_SYSTEM},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, date_string=self.date_string)

    @torch.no_grad()
    def _yes_probability(self, prompts):
        """P(Yes) / (P(Yes) + P(No)) at the first generated position, one forward."""
        yes_id, no_id = self._yes_no_ids()
        self.tokenizer.padding_side = "left"
        # the chat template already carries BOS, and truncation would cut the
        # question off the right end, so neither is applied here
        encoded = self.tokenizer(prompts, padding="longest", return_tensors="pt",
                                 return_token_type_ids=False, add_special_tokens=False)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        outputs = None
        if self._logits_to_keep:
            try:
                outputs = self.model(**encoded, logits_to_keep=1)
            except TypeError:
                self._logits_to_keep = False
        if outputs is None:
            outputs = self.model(**encoded)
        logits = outputs.logits[:, -1, :]
        pair = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=-1).float()
        return torch.softmax(pair, dim=-1)[:, 0].tolist()

    def _sentence_scores(self, query, qid, passages):
        """P(Yes) per sentence, flattened over `passages`, cached by docid#index."""
        cache = self._score_cache("_sentence_cache", self._sentence_cache_path, "exit_sent",
                                  self.sentences_per_score_batch)
        units, keys = [], []
        for rank, passage in enumerate(passages):
            docid = str(passage.get("id"))
            for index, sentence in enumerate(split_sentences(passage.get("text", ""))):
                units.append((rank, index, sentence))
                keys.append(f"{docid}#{index}")

        if not units:
            return [], []
        hit = cache.get(qid, keys)
        if hit is not None:
            return units, hit

        prompts = [self._sentence_prompt(query, passages[rank], sentence)
                   for rank, _, sentence in units]
        # batch similar lengths together: padding is to the longest prompt in the
        # batch, and sentence lengths vary enough that unsorted batches spend a
        # large part of the forward on padding
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        scores = [0.0] * len(units)
        step = max(1, self.sentences_per_score_batch)
        for start in range(0, len(order), step):
            chunk = order[start:start + step]
            for index, score in zip(chunk, self._yes_probability([prompts[i] for i in chunk])):
                scores[index] = score
        cache.put(qid, keys, scores)
        return units, scores

    def _compress(self, query, qid, pool):
        """Greedily fill the token budget with the highest-scoring sentences.

        Selection is by rank inside the query, not by an absolute threshold: an
        untrained scorer is not calibrated to one, and the paper measures that a
        frozen classifier at tau=0.5 keeps 95-99% of the tokens, i.e. compresses
        nothing. The budget is the measured size of the top-5 context it replaces,
        so the comparison holds tokens fixed and varies only coverage.
        """
        units, scores = self._sentence_scores(query, qid, pool)
        if not units:
            return [], []

        order = sorted(range(len(units)), key=lambda i: scores[i], reverse=True)
        header = len(self.tokenizer("Title: \nPassage: \n",
                                    add_special_tokens=False)["input_ids"])

        used, chosen = 0, {}
        for index in order:
            rank, sentence_index, sentence = units[index]
            cost = len(self.tokenizer(" " + sentence, add_special_tokens=False)["input_ids"])
            if rank not in chosen:
                if self.max_blocks is not None and len(chosen) >= self.max_blocks:
                    continue
                cost += header + len(self.tokenizer(
                    pool[rank].get("title", ""), add_special_tokens=False)["input_ids"])
            if used + cost > self.token_budget:
                continue  # a later, shorter sentence may still fit
            chosen.setdefault(rank, []).append((sentence_index, sentence))
            used += cost

        ranks = sorted(chosen)
        self._blocks.append(len(ranks))
        self._tokens.append(used)
        self._sentences.append(sum(len(v) for v in chosen.values()))
        blocks = [{"id": pool[r].get("id"), "title": pool[r].get("title", ""),
                   "text": " ".join(s for _, s in sorted(chosen[r]))} for r in ranks]
        return blocks, ranks

    # -- ModelRAG interface ---------------------------------------------------

    @property
    def assembly_stats(self):
        n = len(self._blocks)
        if not n:
            return {"slot_order": self.slot_order, "use_proxy": self.use_proxy,
                    "fuse": self.fuse, "k_pool": self.k_pool}
        return {
            "slot_order": self.slot_order,
            "use_proxy": self.use_proxy,
            "compress_from": self.compress_from,
            "avg_blocks": sum(self._blocks) / n,
            "avg_context_tokens": sum(self._tokens) / n,
            "avg_sentences": sum(self._sentences) / n,
            "n": n,
        }

    def reset_stats(self):
        self._blocks = []
        self._tokens = []
        self._sentences = []

    def search(self, queries, qids, k=5):
        list_passages, list_scores = [], []
        for (passages, scores), query, qid in zip(self.ranked_pool(queries, qids),
                                                  queries, [str(q) for q in qids]):
            if self.compress_from:
                chosen, ranks = self._compress(query, qid, passages[:self.compress_from])
                chosen_scores = [scores[r] for r in ranks]
            else:
                chosen = [self._strip(p) for p in passages[:k]]
                chosen_scores = scores[:k]
            order = slot_permutation(self.slot_order, len(chosen))
            list_passages.append([chosen[i] for i in order])
            list_scores.append([chosen_scores[i] for i in order])
        return list_passages, list_scores


class CrossEncoderRerankRAG(ImprovedRAG):
    """Rerank with a trained cross-encoder instead of a query-likelihood score.

    This is InfoGain-RAG's inference path: a small model that was distilled from
    DIG labels reads (title, passage, question) and emits one scalar, so scoring a
    candidate costs a single short forward through a 63.58M network rather than a
    250-token teacher-forced pass through the 1.2B generator.
    """

    scorer_id = "cross_encoder"

    def __init__(self, reranker_path=None, cross_max_len=320, **kwargs):
        super().__init__(**kwargs)
        self.reranker_path = reranker_path
        self.cross_max_len = cross_max_len

    def _cross_prompt(self, query, passage):
        text = passage.get("text", "")
        ids = self._st(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.score_max_passage_tokens:
            text = self._st.decode(ids[:self.score_max_passage_tokens],
                                   clean_up_tokenization_spaces=False)
        return (f"Title: {passage.get('title', '')}\nPassage: {text}\n"
                f"Question: {query}")

    @torch.no_grad()
    def score_passages(self, query, passages, qid=None):
        self._st.padding_side = "left"   # the classifier reads the last position
        encoded = self._st([self._cross_prompt(query, p) for p in passages],
                           padding="max_length", max_length=self.cross_max_len,
                           truncation=True, return_tensors="pt",
                           return_token_type_ids=False)
        encoded = {k: v.to(self._sm.device) for k, v in encoded.items()}
        outputs = self._sm(**encoded)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        logits = logits.float()
        return (logits[:, 1] - logits[:, 0]).tolist()
