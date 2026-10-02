"""Build selection targets for the RFT selector from the NQ *train* split.

The selector reads a question and the BM25 candidate pool and emits five indices.
Its reward is how many of the five contain the gold answer string, so the target
used for the optional warm start is simply the reward's argmax: take every
answer-bearing candidate in BM25 order, then pad with the highest-ranked clean
ones so the output is always five indices.

    python prepare_select_train.py            # writes output/results/select_train.jsonl
"""
import io
import json
import os
import re
import string
import sys

POOL = "output/results/bm25_train.jsonl"
OUT = "output/results/select_train.jsonl"
SLOTS = 5

ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCTUATION = set(string.punctuation)


def normalize_answer(text):
    text = "".join(ch for ch in text.lower() if ch not in PUNCTUATION)
    return " ".join(ARTICLES.sub(" ", text).split())


def passage_has_answer(passage, answers):
    blob = normalize_answer("%s %s" % (passage.get("title", ""), passage.get("text", "")))
    return any(answer in blob for answer in answers)


def main():
    if not os.path.exists(POOL):
        raise SystemExit("%s not found; run prepare_dig_train.py bm25 first" % POOL)
    # the reward is built from gold answers, so it may only ever touch the train split
    if "dev" in os.path.basename(POOL):
        raise SystemExit("DEV_GUARD: %s looks like a dev pool" % POOL)

    written = 0
    avail_hist = {}
    handle = io.open(OUT, "w", encoding="utf-8")
    for line in io.open(POOL, encoding="utf-8"):
        row = json.loads(line)
        passages = [p for p in row["passages"] if p.get("text")]
        answers = [normalize_answer(a) for a in row["answers"]]
        answers = [a for a in answers if a]
        if len(passages) < SLOTS or not answers:
            continue

        flags = [passage_has_answer(p, answers) for p in passages]
        gold = [i for i, flag in enumerate(flags) if flag]
        clean = [i for i, flag in enumerate(flags) if not flag]
        target = (gold[:SLOTS] + clean)[:SLOTS]
        target.sort()                       # BM25 order; slot placement is a separate lever

        handle.write(json.dumps({
            "qid": row["qid"],
            "question": row["question"],
            "answers": row["answers"],
            "n_candidates": len(passages),
            "answer_bearing": gold,
            "target": target,
            "reward_ceiling": min(len(gold), SLOTS),
        }, ensure_ascii=False) + "\n")
        written += 1
        key = min(len(gold), 5)
        avail_hist[key] = avail_hist.get(key, 0) + 1
    handle.close()

    print("[select] wrote %s questions to %s" % (format(written, ","), OUT))
    print("[select] answer-bearing candidates available per question")
    for key in sorted(avail_hist):
        label = "5+" if key == 5 else str(key)
        print("           %-3s %7s  (%4.1f%%)"
              % (label, format(avail_hist[key], ","), 100.0 * avail_hist[key] / written))
    reachable = sum(v for k, v in avail_hist.items() if k >= 3)
    print("[select] reward >= 3 reachable on %.1f%% of questions"
          % (100.0 * reachable / written))


if __name__ == "__main__":
    sys.exit(main())
