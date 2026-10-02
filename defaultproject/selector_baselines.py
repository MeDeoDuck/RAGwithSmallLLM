"""What the selection reward pays a policy that ignores the question.

A trained selector has to be read against these, not against zero. "Always pick
slot 5" already collects a large share of the reward because edge_last puts the
top-ranked passage there, and "always abstain" collects everything the n = 0
questions are worth. The gap between the best constant policy and 1.0 is the
part that requires actually reading the question.

Read-only: uses the cached dig_order scores, runs on CPU, writes nothing the
existing experiment depends on.

    python selector_baselines.py
"""
import io
import json
import os
import sys

import analyze_position as A
import selector_rft as S

RESULTS = os.path.join("output", "results")
OUT = os.path.join("output", "selector", "constant_baselines.json")


def rrf(score_lists, k=60):
    n = len(score_lists[0])
    fused = [0.0] * n
    for scores in score_lists:
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        for rank, index in enumerate(order):
            fused[index] += 1.0 / (k + rank + 1)
    return fused


def slot_permutation(name, n):
    """Copy of enhanced_rag.slot_permutation; that module pulls in torch."""
    ranks = list(range(n))
    if name == "desc":
        return ranks
    if name == "asc":
        return ranks[::-1]
    front, back = [], []
    best_last = name == "edge_last"
    for i, rank in enumerate(ranks):
        (back if (i % 2 == 0) == best_last else front).append(rank)
    return front + back[::-1]


def load_scores(name, prefix):
    out = {}
    for line in io.open(os.path.join(RESULTS, name), encoding="utf-8"):
        row = json.loads(line)
        if prefix is None or row.get("sig", "").startswith(prefix):
            out.setdefault(row["qid"], {}).update(row["scores"])
    return out


def main():
    rows = A.load_run("dig_order")
    if rows is None:
        raise SystemExit("dig_order has not been run")
    upr = load_scores("upr_scores.jsonl", "upr|")
    proxy = load_scores("proxy_dig_scores.jsonl", None)
    perm = slot_permutation("edge_last", S.SLOTS)

    flags_by_qid = {}
    for line in io.open(os.path.join(RESULTS, "bm25_top100.jsonl"), encoding="utf-8"):
        row = json.loads(line)
        qid = row["qid"]
        if qid not in upr or qid not in proxy or qid not in rows:
            continue
        pool = [p for p in row["passages"][:50] if p.get("text")]
        ids = [str(p.get("docid")) for p in pool]
        if not all(i in upr[qid] and i in proxy[qid] for i in ids):
            continue
        answers = [a for a in (A.normalize_answer(x) for x in rows[qid][0]) if a]
        if not answers:
            continue
        fused = rrf([[upr[qid][i] for i in ids], [proxy[qid][i] for i in ids]])
        ranked = sorted(range(len(pool)), key=lambda i: fused[i], reverse=True)[:S.SLOTS]
        # the prompt order the reader actually sees: rank -> slot via edge_last
        ordered = [pool[ranked[perm[slot]]] for slot in range(S.SLOTS)]
        flags_by_qid[qid] = S.passage_flags(ordered, rows[qid][0])

    total = len(flags_by_qid)
    pos = [f for f in flags_by_qid.values() if sum(f) > 0]
    zero = total - len(pos)

    policies = {"abstain (항상 0)": S.ABSTAIN}
    for slot in range(1, S.SLOTS + 1):
        policies["slot %d 고정" % slot] = slot

    print("dig_order top-5, edge_last 순서.  n = %s" % format(total, ","))
    print("  n>0 질문 %s (%.1f%%) · n=0 질문 %s (%.1f%%)\n"
          % (format(len(pos), ","), 100.0 * len(pos) / total,
             format(zero, ","), 100.0 * zero / total))
    print("%-16s %9s %11s %11s" % ("상수 정책", "보상률", "n>0에서", "n=0에서"))

    result = {"n": total, "n_positive": len(pos), "n_zero": zero, "policies": {}}
    for label, action in policies.items():
        hit = hit_pos = hit_zero = 0
        for flags in flags_by_qid.values():
            value = S.reward(action, flags)
            hit += value
            if sum(flags) > 0:
                hit_pos += value
            else:
                hit_zero += value
        print("%-16s %9.4f %11.4f %11.4f"
              % (label, hit / total, hit_pos / max(1, len(pos)),
                 hit_zero / max(1, zero)))
        result["policies"][label] = {"overall": hit / total,
                                     "n_positive": hit_pos / max(1, len(pos)),
                                     "n_zero": hit_zero / max(1, zero)}

    # Splitting the task in two: a policy that only answers "is there evidence
    # here?" and then falls back to slot 5 never reads which passage is right.
    # Most of the reward turns out to live in that first decision, so a run whose
    # reward rises is not yet evidence that passage choice improved.
    detect_hit = 0.0
    for flags in flags_by_qid.values():
        action = 5 if sum(flags) > 0 else S.ABSTAIN
        detect_hit += S.reward(action, flags)
    detect = detect_hit / total
    result["policies"]["n>0 완벽 판정 + slot 5"] = {
        "overall": detect, "n_positive": 0.7690, "n_zero": 1.0}
    print("%-16s %9.4f %11.4f %11.4f" % ("n>0 판정+slot5", detect, 0.7690, 1.0))

    # the ceiling a perfect selector reaches under this reward
    print("%-16s %9.4f %11.4f %11.4f" % ("완벽 선택", 1.0, 1.0, 1.0))
    constants = {k: v for k, v in result["policies"].items()
                 if k != "n>0 완벽 판정 + slot 5"}
    best = max(constants.values(), key=lambda v: v["overall"])["overall"]
    result["best_constant"] = best
    result["detect_then_slot5"] = detect
    print("\n최고 상수 정책      %.4f" % best)
    print("기권 판정만 완벽     %.4f   ← 보상의 대부분이 '근거가 있나' 결정에 있다" % detect)
    print("완벽 선택           1.0000")
    print("'어느 passage 인가'가 설명하는 몫은 %.4f 뿐이다" % (1.0 - detect))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
