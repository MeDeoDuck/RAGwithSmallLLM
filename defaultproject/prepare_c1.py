"""The gold-SFT control, written in the rollout format so the same trainer runs it.

C1 answers the question "is the reward doing anything, or would plain supervision
on the obvious target get there too?". Each question contributes exactly one
candidate -- the gold answer, citing the highest-ranked passage that carries a
gold alias -- with weight 1, which reduces the weighted objective to ordinary SFT.

Questions and contexts are the same file the RFT rollouts are drawn from, so the
two differ in the training signal and the budget, not in the data. The budget
difference is real and is not corrected for: RFT samples and runs several rounds,
C1 is one pass.

    python prepare_c1.py
"""
import io
import json
import os
import sys

TRAIN_PATH = os.path.join("output", "results", "reader_rft_train.jsonl")
OUT_PATH = os.path.join("output", "results", "rollouts_c1.jsonl")


def main():
    if not os.path.exists(TRAIN_PATH):
        raise SystemExit("%s not found; run prepare_reader_rft.py first" % TRAIN_PATH)

    written, skipped = 0, 0
    handle = io.open(OUT_PATH, "w", encoding="utf-8")
    for line in io.open(TRAIN_PATH, encoding="utf-8"):
        row = json.loads(line)
        if row["m"] <= 0:
            skipped += 1
            continue
        source = row["flags"].index(1) + 1          # best-ranked answer-bearing slot
        answer = row["answers"][0]
        target = json.dumps({"answer": answer, "source": source})
        handle.write(json.dumps({
            "qid": row["qid"], "question": row["question"], "answers": row["answers"],
            "passages": row["passages"], "flags": row["flags"], "m": row["m"],
            "candidates": [{"text": target, "weight": 1.0, "reward": None,
                            "source": source, "g": 1}],
        }, ensure_ascii=False) + "\n")
        written += 1
    handle.close()
    print("[c1] wrote %s questions to %s (skipped %s)"
          % (format(written, ","), OUT_PATH, format(skipped, ",")))


if __name__ == "__main__":
    sys.exit(main())
