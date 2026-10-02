"""CPU-only retrieval diagnostics for handout section 4.3 (no LLM calls).

Run before spending GPU time: it measures how much headroom a reranker can
possibly have, and decides between reranking and query rewriting.

    python rag_diagnostics.py pool     --k 100          # one deep BM25 pass, cached to disk
    python rag_diagnostics.py backfill --upto 100       # fill in text for cached ranks beyond --keep-text
    python rag_diagnostics.py recall                    # D1/D2/D4: recall@k, hasanswer@k, rank histogram
    python rag_diagnostics.py breakdown --pred output/results/llm_rag_output.txt   # D3
"""

import argparse
import json
import os
import sys
from collections import Counter

from datasets import Dataset as HF_Dataset

from utils.etc import hit2docdict
from utils.metrics import normalize_answer

DEV_PATH = os.path.join("local_cache", "rag", "data", "nq_open_dpr", "nq_dev")
INDEX_PATH = os.path.join("local_cache", "rag", "lucene-index.wikipedia-dpr-100w.20210120.d1b9e6")
POOL_PATH = os.path.join("output", "results", "bm25_top100.jsonl")
KS = [1, 5, 10, 20, 50, 100]


def load_dev():
    dataset = HF_Dataset.load_from_disk(DEV_PATH)
    # same uid rule as RAGDataset.__getitem__ so results join with llm_rag_output.txt
    if "uid" in dataset.column_names:
        uids = [str(u) for u in dataset["uid"]]
    else:
        uids = [f"eval_{i:04d}" for i in range(len(dataset))]
    return dataset, uids


def build_pool(k, batch_size=64, keep_text=50):
    from pyserini.search.lucene import LuceneSearcher

    dataset, uids = load_dev()
    questions = dataset["question"]
    searcher = LuceneSearcher(index_dir=INDEX_PATH)
    os.makedirs(os.path.dirname(POOL_PATH), exist_ok=True)

    with open(POOL_PATH, "w", encoding="utf-8") as out:
        for start in range(0, len(questions), batch_size):
            chunk_q = questions[start:start + batch_size]
            chunk_id = uids[start:start + batch_size]
            hits = searcher.batch_search(queries=list(chunk_q), qids=list(chunk_id),
                                         k=k, threads=8)
            for qid in chunk_id:
                passages = []
                for rank, hit in enumerate(hits[qid]):
                    doc = hit2docdict(hit)
                    title, sep, text = doc["contents"].partition("\n")
                    if not sep:
                        title, text = "", title
                    record = {"docid": str(hit.docid), "score": float(hit.score)}
                    if rank < keep_text:
                        record["title"] = title.strip().strip('"')
                        record["text"] = text.strip()
                    passages.append(record)
                out.write(json.dumps({"qid": qid, "passages": passages}) + "\n")
            print(f"[pool] {min(start + batch_size, len(questions))}/{len(questions)}", flush=True)
    print(f"[pool] wrote {POOL_PATH}")


def load_pool():
    if not os.path.exists(POOL_PATH):
        sys.exit(f"{POOL_PATH} not found: run `python rag_diagnostics.py pool` first")
    pool = {}
    with open(POOL_PATH, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            pool[record["qid"]] = record["passages"]
    return pool


def _has_answer(passage, answers_norm):
    text = normalize_answer(f"{passage.get('title', '')} {passage.get('text', '')}")
    return any(answer and answer in text for answer in answers_norm)


def recall_report():
    dataset, uids = load_dev()
    pool = load_pool()

    gold_hits = {k: 0 for k in KS}
    answer_hits = {k: 0 for k in KS}
    first_answer_rank = Counter()
    n_text = 0

    for i, qid in enumerate(uids):
        passages = pool.get(qid, [])
        row = dataset[i]
        gold = {str(ctx["passage_id"]) for ctx in row.get("positive_ctxs", [])
                if "passage_id" in ctx}
        answers_norm = [normalize_answer(a) for a in row["answers"]]

        gold_rank = next((r for r, p in enumerate(passages) if p["docid"] in gold), None)
        scored = [p for p in passages if "text" in p]
        n_text = max(n_text, len(scored))
        ans_rank = next((r for r, p in enumerate(scored) if _has_answer(p, answers_norm)), None)
        if ans_rank is not None:
            first_answer_rank[ans_rank + 1] += 1

        for k in KS:
            if gold_rank is not None and gold_rank < k:
                gold_hits[k] += 1
            if ans_rank is not None and ans_rank < k:
                answer_hits[k] += 1

    n = len(uids)
    print(f"n = {n} dev questions; answer-presence measured over the first {n_text} passages\n")
    print(f"{'k':>4} {'gold recall@k':>15} {'hasanswer@k':>13}")
    for k in KS:
        gold_note = "" if k <= n_text else "  (text not cached)"
        print(f"{k:>4} {gold_hits[k] / n:>15.4f} "
              f"{answer_hits[k] / n if k <= n_text else float('nan'):>13.4f}{gold_note}")

    print("\nrank of the first answer-bearing passage (top-20):")
    for rank in range(1, 21):
        if first_answer_rank[rank]:
            print(f"  rank {rank:>3}: {first_answer_rank[rank]:>5} ({first_answer_rank[rank] / n:.4f})")

    if n_text >= 50:
        gap = answer_hits[50] / n - answer_hits[5] / n
        print(f"\nhasanswer@50 - hasanswer@5 = {gap:.4f}")
        if gap >= 0.15:
            print("  -> reranking a deeper pool is the right lever (k_pool=20)")
        elif gap >= 0.08:
            print("  -> reranking is worth it, but start from k_pool=50")
        else:
            print("  -> the pool rarely contains the answer: rerank the query, not the passages "
                  "(RRF over query variants) instead of UPR reranking")

    print("\nNote: positive_ctxs in DPR-NQ were themselves selected from BM25 output, so gold "
          "recall@k is biased in favour of BM25. Use hasanswer@k for absolute claims.")


def backfill_text(upto=100, batch_size=500):
    """Fill in title/text for cached ranks that build_pool skipped (rank >= keep_text).

    Reuses the docids/scores already in POOL_PATH -- no BM25 re-search, just a doc
    lookup by id -- so this is much cheaper than re-running `pool` with a larger
    --keep-text. Without this, recall_report() reports hasanswer@k as nan for any
    k beyond the original keep_text cutoff, which must not be confused with a
    measured value of 0 or with the (biased) gold recall@k.
    """
    from pyserini.search.lucene import LuceneSearcher

    if not os.path.exists(POOL_PATH):
        sys.exit(f"{POOL_PATH} not found: run `python rag_diagnostics.py pool` first")

    searcher = LuceneSearcher(index_dir=INDEX_PATH)
    records = [json.loads(line) for line in open(POOL_PATH, encoding="utf-8")]

    missing = sum(1 for r in records for p in r["passages"][:upto] if "text" not in p)
    print(f"[backfill] {missing} passages missing text up to rank {upto}", flush=True)

    filled = 0
    for i, record in enumerate(records):
        for p in record["passages"][:upto]:
            if "text" in p:
                continue
            doc = searcher.doc(p["docid"])
            if doc is None:
                p["title"], p["text"] = "", ""
                continue
            contents = json.loads(doc.raw())["contents"]
            title, sep, text = contents.partition("\n")
            if not sep:
                title, text = "", title
            p["title"] = title.strip().strip('"')
            p["text"] = text.strip()
            filled += 1
        if (i + 1) % batch_size == 0:
            print(f"[backfill] {i + 1}/{len(records)} queries, {filled} passages filled", flush=True)

    with open(POOL_PATH, "w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")
    print(f"[backfill] filled {filled} passages, wrote {POOL_PATH}")


def breakdown(pred_path, k=5):
    if not os.path.exists(pred_path):
        sys.exit(f"{pred_path} not found (run the zero-shot / baseline RAG evaluation first)")

    dataset, uids = load_dev()
    pool = load_pool()
    answers_by_uid = {uid: dataset[i]["answers"] for i, uid in enumerate(uids)}

    buckets = {True: [0, 0], False: [0, 0]}  # has_answer -> [correct, total]
    lengths = []
    with open(pred_path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            uid, _question, _gold, pred = fields[0], fields[1], fields[2], fields[3]
            answers = answers_by_uid.get(uid)
            if answers is None:
                continue
            answers_norm = [normalize_answer(a) for a in answers]
            pred_norm = normalize_answer(pred)
            correct = bool(pred_norm) and any(a and a in pred_norm for a in answers_norm)
            present = any(_has_answer(p, answers_norm)
                          for p in pool.get(uid, [])[:k] if "text" in p)
            buckets[present][1] += 1
            buckets[present][0] += int(correct)
            lengths.append(len(pred.split()))

    total = sum(v[1] for v in buckets.values())
    if not total:
        sys.exit("no rows joined: check that the uid column of the prediction file matches the dev set")
    print(f"joined {total} predictions from {pred_path}\n")
    for present in (True, False):
        correct, count = buckets[present]
        label = f"hasanswer@{k} = {present}"
        rate = correct / count if count else float('nan')
        print(f"{label:<22} n={count:>5}  acc={rate:.4f}")
    overall = sum(v[0] for v in buckets.values()) / total
    print(f"{'overall':<22} n={total:>5}  acc={overall:.4f}")
    print(f"mean prediction length = {sum(lengths) / len(lengths):.2f} words")
    print("\nacc | hasanswer=True is the extraction ceiling: if it is already high, fix retrieval; "
          "if it is low, fix the prompt.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    pool_parser = sub.add_parser("pool", help="one deep BM25 pass, cached to output/results")
    pool_parser.add_argument("--k", type=int, default=100)
    pool_parser.add_argument("--keep-text", type=int, default=50)

    backfill_parser = sub.add_parser("backfill", help="fill cached text for ranks beyond --keep-text")
    backfill_parser.add_argument("--upto", type=int, default=100)

    sub.add_parser("recall", help="recall@k, hasanswer@k and the answer-rank histogram")

    break_parser = sub.add_parser("breakdown", help="accuracy conditioned on answer presence")
    break_parser.add_argument("--pred", default=os.path.join("output", "results", "llm_rag_output.txt"))
    break_parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()
    if args.command == "pool":
        build_pool(args.k, keep_text=args.keep_text)
    elif args.command == "backfill":
        backfill_text(upto=args.upto)
    elif args.command == "recall":
        recall_report()
    else:
        breakdown(args.pred, k=args.k)


if __name__ == "__main__":
    main()
