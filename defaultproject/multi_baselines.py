"""What the Dice reward pays a policy that ignores the question.

The single-selection reward turned out to be dominated by the abstain decision;
this checks whether the multi-selection reward has the same property. Conditions
D (always empty) and E (always passage 5) in the evaluation correspond to two of
the rows here, so a trained policy that fails to beat them has not learned the
task even if its reward looks respectable.

Read-only, CPU, uses the cached dig_order scores.

    python multi_baselines.py
"""
import io
import json
import os
import sys

import analyze_position as A
import multi_select_rft as M
from selector_baselines import load_scores, rrf, slot_permutation

RESULTS = os.path.join("output", "results")
OUT = os.path.join("output", "multi_select", "constant_baselines.json")


def main():
    rows = A.load_run("dig_order")
    if rows is None:
        raise SystemExit("dig_order has not been run")
    upr = load_scores("upr_scores.jsonl", "upr|")
    proxy = load_scores("proxy_dig_scores.jsonl", None)
    perm = slot_permutation("edge_last", M.SLOTS)

    flags_all = []
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
        ranked = sorted(range(len(pool)), key=lambda i: fused[i], reverse=True)[:M.SLOTS]
        ordered = [pool[ranked[perm[slot]]] for slot in range(M.SLOTS)]
        flags_all.append(M.passage_flags(ordered, rows[qid][0]))

    total = len(flags_all)
    positive = [f for f in flags_all if sum(f) > 0]
    zero = total - len(positive)

    policies = [
        ("항상 빈 목록 []", lambda f: [], "조건 D"),
        ("항상 [5]", lambda f: [5], "조건 E"),
        ("항상 전부 [1..5]", lambda f: [1, 2, 3, 4, 5], "조건 A 와 같은 입력"),
        ("n>0 판정 + 전부", lambda f: [1, 2, 3, 4, 5] if sum(f) else [], ""),
        ("n>0 판정 + [5]", lambda f: [5] if sum(f) else [], ""),
        ("완벽 선택 (= T)", lambda f: sorted(M.label_set(f)), "보상 만점"),
    ]

    print("dig_order top-5, edge_last.  n = %s  (n>0 %s / n=0 %s)\n"
          % (format(total, ","), format(len(positive), ","), format(zero, ",")))
    print("%-20s %10s %10s %10s  %s" % ("정책", "평균 보상", "n>0", "n=0", "대응 조건"))
    result = {"n": total, "n_positive": len(positive), "n_zero": zero, "policies": {}}
    for label, fn, note in policies:
        overall = pos_sum = zero_sum = 0.0
        for flags in flags_all:
            value = M.reward(fn(flags), flags)
            overall += value
            if sum(flags):
                pos_sum += value
            else:
                zero_sum += value
        row = {"overall": overall / total,
               "n_positive": pos_sum / max(1, len(positive)),
               "n_zero": zero_sum / max(1, zero)}
        result["policies"][label] = row
        print("%-20s %10.4f %10.4f %10.4f  %s"
              % (label, row["overall"], row["n_positive"], row["n_zero"], note))

    constants = [result["policies"][k]["overall"]
                 for k in ("항상 빈 목록 []", "항상 [5]", "항상 전부 [1..5]")]
    detect = result["policies"]["n>0 판정 + 전부"]["overall"]
    result["best_constant"] = max(constants)
    result["detect_then_all"] = detect
    print("\n최고 상수 정책        %.4f" % max(constants))
    print("기권 판정만 완벽 + 전부  %.4f" % detect)
    print("완벽 선택             1.0000")
    print("'어느 passage 인가' 가 설명하는 몫 %.4f" % (1.0 - detect))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
