"""Split the dev result of an RFT configuration by m, and test it against controls.

Overall accuracy mixes the 39.3% of questions whose prompt never held the answer
with the ones this experiment is about, so it cannot be the headline. The split
below is the reporting contract from the plan:

    overall / m>0 / m=0 / format validity / source hit rate

`m` is recomputed here from the cached scores rather than read from the run, so
it is the same quantity the training filter used.

    python analyze_rft.py rft_g1 rft_c1 rft_r1 rft_r2 rft_r3
"""
import io
import json
import os
import sys

import analyze_position as A
from reader_rft import parse_output

RESULTS = os.path.join("output", "results")
BASELINE = "dig_order"


def rrf(score_lists, k=60):
    n = len(score_lists[0])
    fused = [0.0] * n
    for scores in score_lists:
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        for rank, index in enumerate(order):
            fused[index] += 1.0 / (k + rank + 1)
    return fused


def load_scores(name, prefix):
    out = {}
    for line in io.open(os.path.join(RESULTS, name), encoding="utf-8"):
        row = json.loads(line)
        if prefix is None or row.get("sig", "").startswith(prefix):
            out.setdefault(row["qid"], {}).update(row["scores"])
    return out


def context_flags(answers_by_qid):
    """{qid: [g_1..g_5]} for the dig_order prompt, from the cached scores."""
    upr = load_scores("upr_scores.jsonl", "upr|")
    proxy = load_scores("proxy_dig_scores.jsonl", None)
    flags = {}
    for line in io.open(os.path.join(RESULTS, "bm25_top100.jsonl"), encoding="utf-8"):
        row = json.loads(line)
        qid = row["qid"]
        if qid not in upr or qid not in proxy or qid not in answers_by_qid:
            continue
        pool = [p for p in row["passages"][:50] if p.get("text")]
        ids = [str(p.get("docid")) for p in pool]
        if not all(i in upr[qid] and i in proxy[qid] for i in ids):
            continue
        fused = rrf([[upr[qid][i] for i in ids], [proxy[qid][i] for i in ids]])
        top = sorted(range(len(pool)), key=lambda i: fused[i], reverse=True)[:5]
        answers = [a for a in (A.normalize_answer(x) for x in answers_by_qid[qid]) if a]
        if not answers:
            continue
        flags[qid] = [int(any(a in A.normalize_answer(
            "%s %s" % (pool[i].get("title", ""), pool[i].get("text", "")))
            for a in answers)) for i in top]
    return flags


def load_raw(name):
    """{qid: raw_generation} -- the source field only survives in the raw text."""
    path = os.path.join(RESULTS, "prompt_rag_%s_output.txt" % name)
    if not os.path.exists(path):
        return None
    out = {}
    for line in io.open(path, encoding="utf-8"):
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 5:
            out[fields[0]] = fields[4]
    return out


def summarise(name, correct, flags, raw):
    rows = {"n": 0, "hit": 0, "pres_n": 0, "pres_hit": 0, "abs_n": 0, "abs_hit": 0,
            "valid": 0, "src_n": 0, "src_hit": 0}
    for qid, g in flags.items():
        if qid not in correct:
            continue
        rows["n"] += 1
        rows["hit"] += correct[qid]
        if sum(g) > 0:
            rows["pres_n"] += 1
            rows["pres_hit"] += correct[qid]
        else:
            rows["abs_n"] += 1
            rows["abs_hit"] += correct[qid]
        if raw is not None and qid in raw:
            _, source, valid = parse_output(raw[qid])
            rows["valid"] += int(valid)
            if valid and sum(g) > 0:
                rows["src_n"] += 1
                rows["src_hit"] += g[source - 1]
    return rows


def main():
    names = sys.argv[1:]
    if not names:
        raise SystemExit(__doc__)

    base_rows = A.load_run(BASELINE)
    if base_rows is None:
        raise SystemExit("%s has not been run" % BASELINE)
    flags = context_flags({q: v[0] for q, v in base_rows.items()})
    base_correct = A.correct_map(base_rows)

    table = []
    for name in names:
        rows = A.load_run(name)
        if rows is None:
            print("[skip] %s has not been run" % name)
            continue
        table.append((name, A.correct_map(rows),
                      summarise(name, A.correct_map(rows), flags, load_raw(name))))

    print("%-10s %8s %9s %9s %9s %9s %9s"
          % ("설정", "n", "전체", "m>0", "m=0", "형식유효", "source적중"))
    base_stat = summarise(BASELINE, base_correct, flags, None)
    print("%-10s %8s %9.4f %9.4f %9.4f %9s %9s"
          % (BASELINE, format(base_stat["n"], ","),
             base_stat["hit"] / base_stat["n"],
             base_stat["pres_hit"] / base_stat["pres_n"],
             base_stat["abs_hit"] / base_stat["abs_n"], "—", "—"))
    for name, _, s in table:
        print("%-10s %8s %9.4f %9.4f %9.4f %9s %9s"
              % (name, format(s["n"], ","), s["hit"] / s["n"],
                 s["pres_hit"] / max(1, s["pres_n"]),
                 s["abs_hit"] / max(1, s["abs_n"]),
                 "%.4f" % (s["valid"] / s["n"]) if s["valid"] else "—",
                 "%.4f" % (s["src_hit"] / s["src_n"]) if s["src_n"] else "—"))

    print("\n페어드 McNemar (전체 dev)")
    print("%-24s %9s %9s %7s %10s" % ("비교", "acc_a", "acc_b", "z", "p"))
    for name, correct, _ in table:
        shared = sorted(set(base_correct) & set(correct))
        z, p, b01, b10 = A.mcnemar(base_correct, correct, shared)
        acc_a = sum(base_correct[q] for q in shared) / len(shared)
        acc_b = sum(correct[q] for q in shared) / len(shared)
        print("%-24s %9.4f %9.4f %7.2f %10.4g   win %s lose %s"
              % ("%s → %s" % (BASELINE, name), acc_a, acc_b, z, p,
                 format(b01, ","), format(b10, ",")))

    g1 = next((c for n, c, _ in table if n.endswith("g1")), None)
    if g1 is not None:
        print("\n주 비교 — 같은 출력 형식끼리 (학습 효과만)")
        for name, correct, _ in table:
            if name.endswith("g1"):
                continue
            shared = sorted(set(g1) & set(correct))
            z, p, b01, b10 = A.mcnemar(g1, correct, shared)
            acc_a = sum(g1[q] for q in shared) / len(shared)
            acc_b = sum(correct[q] for q in shared) / len(shared)
            print("%-24s %9.4f %9.4f %7.2f %10.4g   win %s lose %s"
                  % ("rft_g1 → %s" % name, acc_a, acc_b, z, p,
                     format(b01, ","), format(b10, ",")))


if __name__ == "__main__":
    sys.exit(main())
