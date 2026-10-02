"""Selector training contexts from the NQ *train* split, n = 0 included.

The reward is dominated by the abstain decision -- a policy that only answers
"is there evidence here?" and then always picks slot 5 already scores 0.8597 of
1.0 on dev. So the share of n = 0 questions is not a detail: train on too many
and the policy learns to abstain, train on too few and it learns never to.
The target below is the measured dev `dig_order` distribution, n = 0 and all.

Train questions have no UPR or ALR scores cached, so the five passages are drawn
from the BM25 top-20 instead of from the fused ranking. That is a difference
from the evaluation contexts and is reported rather than papered over.

    python prepare_selector_train.py --questions 6000
"""
import argparse
import io
import json
import os
import random
import sys

from selector_rft import SLOTS, passage_flags

POOL_PATH = os.path.join("output", "results", "bm25_train.jsonl")
OUT_PATH = os.path.join("output", "results", "selector_train.jsonl")

# dev dig_order, all 6,514 questions: share of each n
N_TARGET = {0: 2558 / 6514, 1: 1204 / 6514, 2: 910 / 6514,
            3: 649 / 6514, 4: 512 / 6514, 5: 681 / 6514}

# dev dig_order under edge_last: P(slot holds an answer-bearing passage | n)
SLOT_MARGINAL = {
    1: [0.154, 0.097, 0.108, 0.127, 0.513],
    2: [0.626, 0.186, 0.159, 0.277, 0.752],
    3: [0.798, 0.361, 0.297, 0.661, 0.883],
    4: [0.922, 0.744, 0.535, 0.850, 0.949],
    5: [1.000, 1.000, 1.000, 1.000, 1.000],
}


def weighted_pick(options, weights, rng):
    total = sum(weights)
    if total <= 0:
        return rng.choice(options)
    draw, cumulative = rng.random() * total, 0.0
    for option, weight in zip(options, weights):
        cumulative += weight
        if draw <= cumulative:
            return option
    return options[-1]


def choose_slots(n, rng):
    weights = list(SLOT_MARGINAL[n])
    slots, pool = [], list(range(SLOTS))
    for _ in range(n):
        total = sum(weights[i] for i in pool)
        pick = pool[-1]
        if total > 0:
            draw, cumulative = rng.random() * total, 0.0
            for i in pool:
                cumulative += weights[i]
                if draw <= cumulative:
                    pick = i
                    break
        else:
            pick = rng.choice(pool)
        slots.append(pick)
        pool.remove(pick)
    return sorted(slots)


def assign_n(bounds, quota, rng):
    """Fill the target histogram; most-constrained questions choose first."""
    remaining = dict(quota)
    chosen = {}
    for index in sorted(range(len(bounds)), key=lambda i: bounds[i][1]):
        low, high = bounds[index]
        options = [n for n in range(low, high + 1) if n in N_TARGET]
        if not options:
            continue
        left = [max(0.0, remaining.get(n, 0)) for n in options]
        weights = left if sum(left) > 0 else [N_TARGET[n] for n in options]
        n = weighted_pick(options, weights, rng)
        remaining[n] = remaining.get(n, 0) - 1
        chosen[index] = n
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    if not os.path.exists(POOL_PATH):
        raise SystemExit("%s not found" % POOL_PATH)
    if "train" not in os.path.basename(POOL_PATH):
        raise SystemExit("DEV_GUARD: %s is not a train pool" % POOL_PATH)

    rng = random.Random(args.seed)
    usable, bounds, skipped = [], [], 0
    for line in io.open(POOL_PATH, encoding="utf-8"):
        row = json.loads(line)
        pool = [p for p in row["passages"] if p.get("text")]
        if len(pool) < SLOTS:
            skipped += 1
            continue
        flags = passage_flags(pool, row["answers"])
        gold = [i for i, g in enumerate(flags) if g]
        clean = [i for i, g in enumerate(flags) if not g]
        # unlike the reader experiment, n = 0 is a training target here
        low, high = max(0, SLOTS - len(clean)), min(SLOTS, len(gold))
        if low > high:
            skipped += 1
            continue
        usable.append((row, pool, flags, gold, clean))
        bounds.append((low, high))
        if len(usable) >= args.questions:
            break

    quota = {n: len(usable) * share for n, share in N_TARGET.items()}
    chosen = assign_n(bounds, quota, rng)

    written, n_hist, slot_hist = 0, {}, [0] * SLOTS
    handle = io.open(args.out, "w", encoding="utf-8")
    for index, (row, pool, flags, gold, clean) in enumerate(usable):
        n = chosen.get(index)
        if n is None:
            continue
        ordered = [None] * SLOTS
        if n:
            slots = choose_slots(n, rng)
            for slot, pool_index in zip(slots, gold[:n]):
                ordered[slot] = pool_index
            for slot in slots:
                slot_hist[slot] += 1
        spare = iter(clean[:SLOTS - n])
        for slot in range(SLOTS):
            if ordered[slot] is None:
                ordered[slot] = next(spare)

        passages = [{"id": pool[i].get("docid"), "title": pool[i].get("title", ""),
                     "text": pool[i]["text"]} for i in ordered]
        placed = [flags[i] for i in ordered]
        handle.write(json.dumps({
            "qid": row["qid"], "question": row["question"], "answers": row["answers"],
            "passages": passages, "flags": placed, "n": sum(placed),
        }, ensure_ascii=False) + "\n")
        written += 1
        n_hist[n] = n_hist.get(n, 0) + 1
    handle.close()

    print("[sel] wrote %s questions to %s (skipped %s)"
          % (format(written, ","), args.out, format(skipped, ",")))
    print("\n[sel] n 분포 — 생성 vs dev dig_order")
    print("   %3s %9s %9s %9s" % ("n", "문항", "생성", "목표"))
    for n in sorted(N_TARGET):
        got = n_hist.get(n, 0)
        print("   %3d %9s %8.1f%% %8.1f%%"
              % (n, format(got, ","), 100.0 * got / written, 100.0 * N_TARGET[n]))
    print("\n[sel] 정답 보유 passage 칸 분포 — 생성 vs dev 실측")
    eval_overall = [0.613, 0.400, 0.360, 0.493, 0.769]
    positive = sum(c for n, c in n_hist.items() if n > 0) or 1
    for slot in range(SLOTS):
        print("   칸 %d  생성 %.3f   dev %.3f"
              % (slot + 1, slot_hist[slot] / positive, eval_overall[slot]))
    print("\n[sel] 주의: train 컨텍스트는 BM25 top-20 에서 구성했다. "
          "평가는 UPR+ALR RRF 융합 top-5 이므로 passage 품질이 같지 않다.")


if __name__ == "__main__":
    sys.exit(main())
