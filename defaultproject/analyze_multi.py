"""Selection quality and answer accuracy for the multi-selection conditions.

Two things are reported side by side and never merged. Selection quality is what
the reward optimised; answer accuracy is what the hypothesis is about. If the
first does not move and the second does, the gain did not come from better
evidence selection, and the table has to make that visible.

Denominators, stated because they differ:
  * every rate over "전체" divides by all evaluated questions, malformed
    outputs included -- a format error is not a correct empty selection
  * precision is undefined when the model selects nothing (no denominator); it
    is dropped from the precision mean rather than scored 0 or 1, and the count
    of dropped questions is printed
  * recall and F1 are 0 for an empty selection when n > 0
  * n = 0 questions have no F1; they are scored by whether the model abstained

McNemar is applied to answer correctness only. Selection F1 is continuous, so a
discordant-pair test does not apply to it.

    python analyze_multi.py msel_a msel_b msel_c msel_d msel_e
"""
import io
import json
import math
import os
import sys

import analyze_position as A
import multi_select_rft as M

RESULTS = os.path.join("output", "results")
LOG_DIR = os.path.join("output", "multi_select")


def load_log(name):
    path = os.path.join(LOG_DIR, "%s_selection.jsonl" % name)
    if not os.path.exists(path):
        return None
    return {row["qid"]: row for row in
            (json.loads(line) for line in io.open(path, encoding="utf-8"))}


def summarise(log, correct):
    agg = {
        "n": 0, "valid": 0, "answer_hit": 0,
        "f1_sum": 0.0, "f1_n": 0,              # n > 0 only
        "prec_sum": 0.0, "prec_n": 0, "prec_undefined": 0,
        "rec_sum": 0.0, "exact": 0, "exact_n": 0,
        "k_sum": 0, "chars_sum": 0,
        "pos_n": 0, "pos_hit": 0, "pos_empty": 0,
        "zero_n": 0, "zero_hit": 0, "zero_abstain": 0,
        "by_n": {}, "by_k": {},
    }
    for qid, row in log.items():
        if qid not in correct:
            continue
        agg["n"] += 1
        agg["valid"] += int(row["valid"])
        hit = correct[qid]
        agg["answer_hit"] += hit
        agg["k_sum"] += len(row["sources"])
        agg["chars_sum"] += row.get("picked_chars", 0)
        scores = M.selection_scores(row["sources"], row["flags"])
        n = scores["n"]
        bucket = agg["by_n"].setdefault(n, {"q": 0, "f1": 0.0, "ans": 0, "exact": 0})
        bucket["q"] += 1
        bucket["ans"] += hit
        bucket["exact"] += int(scores["exact"])
        kb = agg["by_k"].setdefault(len(row["sources"]), {"q": 0, "ans": 0})
        kb["q"] += 1
        kb["ans"] += hit
        agg["exact_n"] += 1
        agg["exact"] += int(scores["exact"])
        if n > 0:
            agg["pos_n"] += 1
            agg["pos_hit"] += hit
            agg["pos_empty"] += int(not row["sources"])
            agg["f1_sum"] += scores["f1"]
            agg["f1_n"] += 1
            bucket["f1"] += scores["f1"]
            agg["rec_sum"] += scores["recall"]
            if scores["precision"] is None:
                agg["prec_undefined"] += 1
            else:
                agg["prec_sum"] += scores["precision"]
                agg["prec_n"] += 1
        else:
            agg["zero_n"] += 1
            agg["zero_hit"] += hit
            agg["zero_abstain"] += int(not row["sources"])
    return agg


def row_for(name, agg):
    total = max(1, agg["n"])
    pos = max(1, agg["pos_n"])
    zero = max(1, agg["zero_n"])
    return [
        name, format(agg["n"], ","),
        "%.4f" % (agg["answer_hit"] / total),
        "%.4f" % (agg["pos_hit"] / pos),
        "%.4f" % (agg["zero_hit"] / zero),
        "%.4f" % (agg["f1_sum"] / max(1, agg["f1_n"])),
        "%.4f" % (agg["prec_sum"] / max(1, agg["prec_n"])),
        "%.4f" % (agg["rec_sum"] / pos),
        "%.4f" % (agg["exact"] / max(1, agg["exact_n"])),
        "%.4f" % (agg["zero_abstain"] / zero),
        "%.4f" % (agg["pos_empty"] / pos),
        "%.4f" % (agg["valid"] / total),
        "%.2f" % (agg["k_sum"] / total),
    ]


def main():
    names = sys.argv[1:]
    if not names:
        raise SystemExit(__doc__)

    table, logs, corrects = [], {}, {}
    for name in names:
        log = load_log(name)
        rows = A.load_run(name)
        if log is None or rows is None:
            print("[skip] %s — 로그 또는 출력이 없다" % name)
            continue
        correct = A.correct_map(rows)
        logs[name], corrects[name] = log, correct
        table.append((name, summarise(log, correct)))

    header = ["설정", "n", "답변전체", "답변n>0", "답변n=0", "선택F1",
              "정밀도", "재현율", "완전일치", "n=0기권", "n>0빈선택", "형식유효", "평균k"]
    widths = [10, 7, 8, 8, 8, 7, 7, 7, 8, 8, 9, 8, 6]
    print(" ".join(h.rjust(w) for h, w in zip(header, widths)))
    for name, agg in table:
        print(" ".join(c.rjust(w) for c, w in zip(row_for(name, agg), widths)))

    for name, agg in table:
        if agg["prec_undefined"]:
            print("\n[%s] 정밀도 분모 없음(빈 선택) %s문항은 정밀도 평균에서 제외. "
                  "재현율·F1 에서는 0 으로 셈." % (name, format(agg["prec_undefined"], ",")))

    print("\n── n 별 선택·답변")
    for name, agg in table:
        parts = []
        for n in sorted(agg["by_n"]):
            b = agg["by_n"][n]
            f1 = ("%.3f" % (b["f1"] / b["q"])) if n > 0 else "  — "
            parts.append("n=%d(%s) F1 %s acc %.3f" % (n, format(b["q"], ","), f1,
                                                      b["ans"] / b["q"]))
        print("  %-10s %s" % (name, " | ".join(parts)))

    print("\n── 선택 개수별 답변 정확도")
    for name, agg in table:
        parts = ["k=%d(%s) %.3f" % (k, format(v["q"], ","), v["ans"] / v["q"])
                 for k, v in sorted(agg["by_k"].items())]
        print("  %-10s %s" % (name, " | ".join(parts)))

    # paired comparison on answer correctness only
    base = next((n for n in ("msel_b",) if n in corrects), None)
    if base:
        print("\n── 대응표본 McNemar (답변 정오만. 선택 F1 에는 적용하지 않음)")
        print("%-22s %9s %9s %7s %10s" % ("비교", "acc_a", "acc_b", "z", "p"))
        for name in corrects:
            if name == base:
                continue
            shared = sorted(set(corrects[base]) & set(corrects[name]))
            z, p, b01, b10 = A.mcnemar(corrects[base], corrects[name], shared)
            acc_a = sum(corrects[base][q] for q in shared) / len(shared)
            acc_b = sum(corrects[name][q] for q in shared) / len(shared)
            print("%-22s %9.4f %9.4f %7.2f %10.4g   win %s lose %s"
                  % ("%s → %s" % (base, name), acc_a, acc_b, z, p,
                     format(b01, ","), format(b10, ",")))

    for name in names:
        stats = os.path.join(LOG_DIR, "%s_grpo_stats.json" % name)
        if os.path.exists(stats):
            data = json.load(io.open(stats, encoding="utf-8"))
            print("\n[%s] 영분산 %.1f%%  학습 그룹 %s  형식오류 %.2f%%"
                  % (name, 100 * data.get("zero_variance_rate", 0),
                     format(data.get("used_groups", 0), ","),
                     100 * data.get("malformed_rate", 0)))


if __name__ == "__main__":
    sys.exit(main())
