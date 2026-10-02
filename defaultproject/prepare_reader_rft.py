"""Build the m>0 training contexts for the RFT reader, from the NQ *train* split.

Three distributions have to match the evaluation setting or the policy learns the
construction instead of the task:

  m        how many of the five passages carry a gold alias
  slot     which of the five slots those passages occupy
  content  the distractors are highly-ranked non-answer passages, not random ones

The slot targets are measured from `dig_order` on dev, not assumed. Under
`edge_last` a lone answer-bearing passage sits in slot 5 about half the time --
placing it there always would teach position rather than reading, and placing it
uniformly would not match what the reader meets at evaluation.

    python prepare_reader_rft.py --questions 6000
"""
import argparse
import io
import json
import os
import random
import sys

from reader_rft import SLOTS, passage_flags

POOL_PATH = os.path.join("output", "results", "bm25_train.jsonl")
OUT_PATH = os.path.join("output", "results", "reader_rft_train.jsonl")

# dev `dig_order`, m > 0 questions (n = 3,956): share of each m
M_TARGET = {1: 1204 / 3956, 2: 910 / 3956, 3: 649 / 3956,
            4: 512 / 3956, 5: 681 / 3956}

# dev `dig_order` under edge_last: P(slot holds an answer-bearing passage | m).
# Rows sum to m because m passages are spread over the five slots.
SLOT_MARGINAL = {
    1: [0.154, 0.097, 0.108, 0.127, 0.513],
    2: [0.626, 0.186, 0.159, 0.277, 0.752],
    3: [0.798, 0.361, 0.297, 0.661, 0.883],
    4: [0.922, 0.744, 0.535, 0.850, 0.949],
    5: [1.000, 1.000, 1.000, 1.000, 1.000],
}


def choose_slots(m, rng):
    """m distinct slots drawn with the measured marginals as weights."""
    weights = list(SLOT_MARGINAL[m])
    slots, pool = [], list(range(SLOTS))
    for _ in range(m):
        total = sum(weights[i] for i in pool)
        if total <= 0:
            slots.append(rng.choice(pool))
        else:
            draw = rng.random() * total
            cumulative = 0.0
            pick = pool[-1]
            for i in pool:
                cumulative += weights[i]
                if draw <= cumulative:
                    pick = i
                    break
            slots.append(pick)
        pool.remove(slots[-1])
    return sorted(slots)


def weighted_pick(options, weights, rng):
    total = sum(weights)
    if total <= 0:
        return rng.choice(options)
    draw = rng.random() * total
    cumulative = 0.0
    for option, weight in zip(options, weights):
        cumulative += weight
        if draw <= cumulative:
            return option
    return options[-1]


def assign_m(candidates, quota, rng):
    """Fill the target m histogram instead of sampling each question on its own.

    Drawing from M_TARGET per question overshoots m=1 badly: a question whose
    pool holds only one answer-bearing passage can only be m=1, and questions
    that could support more still pick m=1 most of the time because M_TARGET
    favours it. Handing out the scarce high-m slots to the questions that can
    actually take them, most-constrained first, is what makes the global
    histogram land on the target.
    """
    remaining = dict(quota)
    order = sorted(range(len(candidates)), key=lambda i: candidates[i][1])  # by `high`
    chosen = {}
    for index in order:
        low, high = candidates[index]
        options = [m for m in range(low, high + 1) if m in M_TARGET]
        if not options:
            continue
        left = [max(0.0, remaining.get(m, 0)) for m in options]
        m = weighted_pick(options, left, rng) if sum(left) > 0 else \
            weighted_pick(options, [M_TARGET[m] for m in options], rng)
        remaining[m] = remaining.get(m, 0) - 1
        chosen[index] = m
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    if not os.path.exists(POOL_PATH):
        raise SystemExit("%s not found" % POOL_PATH)
    # gold answers drive the reward, so this may only ever read the train split
    if "train" not in os.path.basename(POOL_PATH):
        raise SystemExit("DEV_GUARD: %s is not a train pool" % POOL_PATH)

    rng = random.Random(args.seed)
    skipped = {"짧은 pool": 0, "정답 보유 0": 0, "clean 부족": 0}

    # pass 1 -- which questions are usable, and what m can each one support
    usable, bounds = [], []
    for line in io.open(POOL_PATH, encoding="utf-8"):
        row = json.loads(line)
        pool = [p for p in row["passages"] if p.get("text")]
        if len(pool) < SLOTS:
            skipped["짧은 pool"] += 1
            continue
        flags = passage_flags(pool, row["answers"])
        gold = [i for i, g in enumerate(flags) if g]
        clean = [i for i, g in enumerate(flags) if not g]
        if not gold:
            skipped["정답 보유 0"] += 1          # m = 0 is not a training target
            continue
        low, high = max(1, SLOTS - len(clean)), min(SLOTS, len(gold))
        if low > high:
            skipped["clean 부족"] += 1
            continue
        usable.append((row, pool, flags, gold, clean))
        bounds.append((low, high))
        if len(usable) >= args.questions:
            break

    target = len(usable)
    quota = {m: target * share for m, share in M_TARGET.items()}
    chosen = assign_m(bounds, quota, rng)

    # pass 2 -- build each context at its assigned m
    written = 0
    m_hist, slot_hist = {}, [0] * SLOTS
    per_m_slots = {m: [0] * SLOTS for m in M_TARGET}
    handle = io.open(args.out, "w", encoding="utf-8")
    for index, (row, pool, flags, gold, clean) in enumerate(usable):
        m = chosen.get(index)
        if m is None:
            skipped["clean 부족"] += 1
            continue
        # highest-ranked candidates of each kind: distractors a ranker would
        # plausibly surface, not arbitrary passages from the tail
        picked_gold, picked_clean = gold[:m], clean[:SLOTS - m]
        slots = choose_slots(m, rng)
        ordered = [None] * SLOTS
        for slot, pool_index in zip(slots, picked_gold):
            ordered[slot] = pool_index
        spare = iter(picked_clean)
        for slot in range(SLOTS):
            if ordered[slot] is None:
                ordered[slot] = next(spare)

        passages = [{"id": pool[i].get("docid"), "title": pool[i].get("title", ""),
                     "text": pool[i]["text"]} for i in ordered]
        placed = [flags[i] for i in ordered]
        handle.write(json.dumps({
            "qid": row["qid"], "question": row["question"], "answers": row["answers"],
            "passages": passages, "flags": placed, "m": sum(placed),
        }, ensure_ascii=False) + "\n")
        written += 1
        m_hist[m] = m_hist.get(m, 0) + 1
        for slot in slots:
            slot_hist[slot] += 1
            per_m_slots[m][slot] += 1
    handle.close()

    print("[rft] wrote %s questions to %s" % (format(written, ","), args.out))
    for reason, count in skipped.items():
        if count:
            print("[rft] skipped %-12s %s" % (reason, format(count, ",")))

    print("\n[rft] m 분포 — 생성된 것 vs dev dig_order 목표")
    print("   %3s %9s %9s %9s" % ("m", "문항", "생성", "목표"))
    for m in sorted(M_TARGET):
        got = m_hist.get(m, 0)
        print("   %3d %9s %8.1f%% %8.1f%%"
              % (m, format(got, ","), 100.0 * got / written, 100.0 * M_TARGET[m]))

    print("\n[rft] 정답 보유 passage의 칸 분포 — 생성 vs dev 실측")
    eval_overall = [0.613, 0.400, 0.360, 0.493, 0.769]
    print("   %5s %9s %9s" % ("칸", "생성", "dev"))
    for slot in range(SLOTS):
        print("   %5d %8.3f %9.3f"
              % (slot + 1, slot_hist[slot] / written, eval_overall[slot]))

    print("\n[rft] m별 칸 분포 — 생성(위) vs dev 목표(아래)")
    for m in sorted(M_TARGET):
        count = m_hist.get(m, 0)
        if not count:
            continue
        got = " ".join("%6.3f" % (per_m_slots[m][s] / count) for s in range(SLOTS))
        want = " ".join("%6.3f" % v for v in SLOT_MARGINAL[m])
        print("   m=%d 생성 %s\n       목표 %s" % (m, got, want))


if __name__ == "__main__":
    sys.exit(main())
