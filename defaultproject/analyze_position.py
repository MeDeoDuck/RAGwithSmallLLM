"""Offline analysis of the position study and the oracle diagnostics (stdlib only).

    python analyze_position.py slots      # slot-order variants vs the desc baseline
    python analyze_position.py goldpos    # controlled gold-slot sweep, injected subset only
    python analyze_position.py dig        # oracle answer-likelihood: ceiling and UPR overlap

Every comparison is paired on question ids, and McNemar's test is reported for the
pairs that disagree, because the differences here are a few points on ~6.5k questions
and an unpaired eyeball comparison cannot tell them from noise.
"""

import argparse
import ast
import io
import json
import math
import os
import re
import string
from collections import defaultdict

RESULTS = os.path.join("output", "results")
POOL_PATH = os.path.join(RESULTS, "bm25_top100.jsonl")
ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)
PUNCTUATION = set(string.punctuation)


def normalize_answer(text):
    text = "".join(ch for ch in text.lower() if ch not in PUNCTUATION)
    return " ".join(ARTICLES_REGEX.sub(" ", text).split())


def load_run(name):
    """{qid: (answers, prediction)} for one configuration, or None if it has not run."""
    path = os.path.join(RESULTS, f"prompt_rag_{name}_output.txt")
    if not os.path.exists(path):
        return None
    rows = {}
    for line in io.open(path, encoding="utf-8"):
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 4:
            rows[fields[0]] = (ast.literal_eval(fields[2]), fields[3])
    return rows


def correct_map(rows):
    """Scores one run, dropping gold aliases that normalise to the empty string.

    best_subspan_exact_match keeps them, and "" is a substring of everything, so it
    counts one dev question correct for every configuration. The absolute accuracy
    here is therefore 1/6515 below the score.json value, identically for every run;
    the paired deltas and McNemar counts below are unaffected.
    """
    out = {}
    for qid, (answers, prediction) in rows.items():
        pred = normalize_answer(prediction)
        out[qid] = int(bool(pred) and any(normalize_answer(a) in pred
                                          for a in answers if normalize_answer(a)))
    return out


def mcnemar(a, b, qids):
    """z and two-sided p for the paired difference b - a over `qids`."""
    b01 = sum(1 for q in qids if not a[q] and b[q])
    b10 = sum(1 for q in qids if a[q] and not b[q])
    if b01 + b10 == 0:
        return 0.0, 1.0, b01, b10
    z = (b01 - b10) / math.sqrt(b01 + b10)
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p, b01, b10


def report(baseline_name, names, subset=None, title=""):
    base = load_run(baseline_name)
    if base is None:
        raise SystemExit(f"{baseline_name} has not been run yet")
    base_correct = correct_map(base)
    runs = {}
    for name in names:
        rows = load_run(name)
        if rows is not None:
            runs[name] = correct_map(rows)
    if not runs:
        raise SystemExit(f"none of {names} has been run yet")

    shared = set(base_correct)
    for correct in runs.values():
        shared &= set(correct)
    if subset is not None:
        shared &= subset
    qids = sorted(shared)
    if not qids:
        raise SystemExit("no questions left after pairing: check that the runs share uids")

    print(f"{title}\nn = {len(qids)} questions, paired; baseline = {baseline_name} "
          f"({sum(base_correct[q] for q in qids) / len(qids):.4f})\n")
    print(f"{'config':<18}{'acc':>9}{'delta':>10}{'z':>8}{'p':>9}{'win':>7}{'lose':>7}")
    print("-" * 68)
    base_acc = sum(base_correct[q] for q in qids) / len(qids)
    for name, correct in runs.items():
        acc = sum(correct[q] for q in qids) / len(qids)
        z, p, win, lose = mcnemar(base_correct, correct, qids)
        print(f"{name:<18}{acc:>9.4f}{acc - base_acc:>+10.4f}{z:>8.2f}{p:>9.4f}{win:>7}{lose:>7}")


def injected_qids(k_pool=50, k=5):
    """Questions GoldPositionRAG could inject: an answer-bearing passage plus k-1 clean ones.

    Neither condition depends on the reranking order, only on which passages are in
    the BM25 pool, so the set is identical for every gold_slot and can be rebuilt here
    without re-running anything.
    """
    any_run = None
    for slot in range(k):
        any_run = load_run(f"goldpos{slot}")
        if any_run is not None:
            break
    if any_run is None:
        raise SystemExit("no goldpos run found")

    out = set()
    for line in io.open(POOL_PATH, encoding="utf-8"):
        record = json.loads(line)
        qid = record["qid"]
        if qid not in any_run:
            continue
        answers_norm = [normalize_answer(a) for a in any_run[qid][0]]
        # same order as UPRRerankRAG._load_cache: drop the text-less ranks first,
        # then take k_pool, or the two would disagree on a pool built with a
        # smaller --keep-text
        with_text = [p for p in record["passages"] if p.get("text")]
        flags = []
        for passage in with_text[:k_pool]:
            body = normalize_answer(f"{passage.get('title', '')} {passage.get('text', '')}")
            flags.append(any(a and a in body for a in answers_norm))
        if any(flags) and sum(1 for f in flags if not f) >= k - 1:
            out.add(qid)
    return out


def spearman(x, y):
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2.0
            for t in range(i, j + 1):
                ranks[order[t]] = shared
            i = j + 1
        return ranks

    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def dig_report():
    dump_path = os.path.join(RESULTS, "oracle_dig_dump.jsonl")
    upr_path = os.path.join(RESULTS, "upr_scores.jsonl")
    if not os.path.exists(dump_path):
        raise SystemExit(f"{dump_path} not found: run the oracle_dig configuration first")

    upr = {}
    if os.path.exists(upr_path):
        for line in io.open(upr_path, encoding="utf-8"):
            record = json.loads(line)
            if record.get("sig", "").startswith("upr|"):
                upr.setdefault(record["qid"], {}).update(record["scores"])

    rhos, negative, total, baseline_beats = [], 0, 0, 0
    seen = set()
    for line in io.open(dump_path, encoding="utf-8"):
        record = json.loads(line)
        qid = record["qid"]
        if qid in seen:
            continue
        seen.add(qid)
        scores, docids, base = record["dig"], record["docids"], record["baseline"]
        total += len(scores)
        negative += sum(1 for s in scores if s - base < 0)
        baseline_beats += int(max(scores) < base)
        hit = upr.get(qid)
        if hit and all(d in hit for d in docids) and len(scores) > 1:
            rho = spearman([hit[d] for d in docids], scores)
            if rho == rho:  # a constant score vector gives nan, which would poison the mean
                rhos.append(rho)

    if not seen or not total:
        raise SystemExit(f"{dump_path} is empty: the oracle_dig run did not produce scores")

    print(f"oracle answer-likelihood vs UPR, {len(seen)} questions\n")
    print(f"passages scored                     {total}")
    print(f"DIG < 0 (passage hurts the answer)  {negative / total:.4f}")
    print(f"no passage beats the no-doc prior   {baseline_beats / len(seen):.4f}")
    if rhos:
        rhos_sorted = sorted(rhos)
        mean = sum(rhos) / len(rhos)
        print(f"\nSpearman(UPR, oracle) per question, over {len(rhos)} questions")
        print(f"  mean   {mean:.4f}")
        print(f"  median {rhos_sorted[len(rhos_sorted) // 2]:.4f}")
        print(f"  p10    {rhos_sorted[int(0.10 * len(rhos_sorted))]:.4f}")
        print(f"  p90    {rhos_sorted[int(0.90 * len(rhos_sorted))]:.4f}")
        print("\nA high correlation means the oracle mostly re-finds what UPR already "
              "ranks highly, so a distilled reranker would have little left to win.")
    else:
        print("\nno UPR scores cached for these questions: run a slot_* configuration first")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["slots", "goldpos", "dig"])
    args = parser.parse_args()

    if args.command == "slots":
        report("ext_rerank_v3_p50", ["slot_edge", "slot_edge_last", "slot_asc", "oracle_dig"],
               title="slot order and oracle reranking, same retrieval, paired")
    elif args.command == "goldpos":
        report("goldpos0", [f"goldpos{s}" for s in range(1, 5)], subset=injected_qids(),
               title="controlled gold-slot sweep (injected questions only)")
    else:
        dig_report()


if __name__ == "__main__":
    main()
