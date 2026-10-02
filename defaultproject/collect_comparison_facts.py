"""Gather everything the three-way comparison report is allowed to cite.

The report must not estimate. This collects what actually exists on disk --
score files, selection logs, GRPO statistics, constant-policy baselines -- and
marks anything absent as not-run rather than leaving a gap that invites a guess.
Each entry carries the path it came from so the report can cite it.

    python collect_comparison_facts.py
"""
import io
import json
import os
import sys

RESULTS = os.path.join("output", "results")
OUT = os.path.join("output", "comparison_facts.json")

# config -> which of the three designs it belongs to
CONFIGS = {
    "dig_order": "baseline",
    "rft_g1": "reader_rft", "rft_c1": "reader_rft",
    "rft_r1": "reader_rft", "rft_r2": "reader_rft", "rft_r3": "reader_rft",
    "sel_oracle1": "single_select", "sel_abstain": "single_select",
    "sel_b": "single_select", "sel_c": "single_select",
    "msel_a": "multi_select", "msel_b": "multi_select", "msel_c": "multi_select",
    "msel_d": "multi_select", "msel_e": "multi_select",
    "msel_oracle": "multi_select",
}

EXTRA_FILES = {
    "single_constant_baselines": os.path.join("output", "selector", "constant_baselines.json"),
    "multi_constant_baselines": os.path.join("output", "multi_select", "constant_baselines.json"),
    "two_call_check": os.path.join("output", "multi_select", "two_call_check.json"),
    "pass_at_8": os.path.join(RESULTS, "rft_pass8.json"),
    "single_grpo_stats": os.path.join("output", "selector", "grpo_r1", "grpo_stats.json"),
    "multi_grpo_stats": os.path.join("output", "multi_select", "grpo_r1", "grpo_stats.json"),
}

# training inputs, so the report can state how each design built them
TRAIN_INPUTS = {
    "reader_rft": (os.path.join(RESULTS, "reader_rft_train.jsonl"),
                   "gold 라벨로 m 분포를 맞춰 재구성한 컨텍스트"),
    "single_select": (os.path.join(RESULTS, "selector_train.jsonl"),
                      "gold 라벨로 n 분포를 맞춰 재구성한 컨텍스트"),
    "multi_select": (os.path.join(RESULTS, "multi_train.jsonl"),
                     "실제 파이프라인(BM25 top-20 → UPR → 1라운드 → ALR → RRF) 출력"),
}


def read_json(path):
    if not os.path.exists(path) or not os.path.getsize(path):
        return None
    try:
        return json.load(io.open(path, encoding="utf-8"))
    except ValueError:
        return None


def count_lines(path):
    if not os.path.exists(path):
        return None
    return sum(1 for _ in io.open(path, encoding="utf-8"))


def selection_summary(name):
    """Per-question selection outcomes, if this config logged them."""
    path = os.path.join("output", "multi_select", "%s_selection.jsonl" % name)
    if not os.path.exists(path):
        return None
    sys.path.insert(0, ".")
    import multi_select_rft as M

    agg = {"n": 0, "valid": 0, "k_sum": 0, "exact": 0,
           "f1_sum": 0.0, "f1_n": 0, "prec_sum": 0.0, "prec_n": 0,
           "prec_undefined": 0, "rec_sum": 0.0,
           "pos_n": 0, "pos_empty": 0, "zero_n": 0, "zero_abstain": 0,
           "by_n": {}}
    for line in io.open(path, encoding="utf-8"):
        row = json.loads(line)
        agg["n"] += 1
        agg["valid"] += int(row["valid"])
        agg["k_sum"] += len(row["sources"])
        scores = M.selection_scores(row["sources"], row["flags"])
        agg["exact"] += int(scores["exact"])
        bucket = agg["by_n"].setdefault(str(scores["n"]), {"q": 0, "f1": 0.0, "exact": 0})
        bucket["q"] += 1
        bucket["exact"] += int(scores["exact"])
        if scores["n"] > 0:
            agg["pos_n"] += 1
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
            agg["zero_abstain"] += int(not row["sources"])
    agg["path"] = path
    return agg


def main():
    facts = {"configs": {}, "missing": [], "files": {}, "train_inputs": {},
             "design_from_code": {}}

    for name, family in CONFIGS.items():
        score_path = os.path.join(RESULTS, "prompt_rag_%s_score.json" % name)
        out_path = os.path.join(RESULTS, "prompt_rag_%s_output.txt" % name)
        data = read_json(score_path)
        if data is None:
            facts["missing"].append({"config": name, "family": family,
                                     "expected": score_path, "status": "미실행"})
            continue
        if data.get("_skipped"):
            # a sentinel left to make the runner skip a stage, not a measurement
            facts["missing"].append({"config": name, "family": family,
                                     "expected": score_path, "status": "의도적 생략",
                                     "reason": data.get("reason")})
            continue
        facts["configs"][name] = {
            "family": family, "score_path": score_path, "output_path": out_path,
            "n_output_rows": count_lines(out_path),
            "metrics": {k: v for k, v in data.items()
                        if isinstance(v, (int, float, str))},
            "parse_stats": data.get("parse_stats"),
            "assembly_stats": data.get("assembly_stats"),
            "selection": selection_summary(name),
        }

    for label, path in EXTRA_FILES.items():
        data = read_json(path)
        facts["files"][label] = {"path": path,
                                 "present": data is not None, "data": data}
        if data is None:
            facts["missing"].append({"artifact": label, "expected": path,
                                     "status": "미실행"})

    for family, (path, note) in TRAIN_INPUTS.items():
        facts["train_inputs"][family] = {
            "path": path, "rows": count_lines(path), "구성": note,
            "present": os.path.exists(path)}

    # design facts read straight out of the modules, not from the plan documents
    try:
        sys.path.insert(0, ".")
        import multi_select_rft as MM
        import reader_rft as RR
        import selector_rft as SS
        facts["design_from_code"] = {
            "reader_rft": {"module": "reader_rft.py",
                           "alpha": RR.ALPHA, "beta": RR.BETA,
                           "answer_in_reward": True,
                           "action": '{"answer": str, "source": 1..5}'},
            "single_select": {"module": "selector_rft.py",
                              "good": SS.REWARD_GOOD, "bad": SS.REWARD_BAD,
                              "malformed": SS.REWARD_MALFORMED,
                              "answer_in_reward": False,
                              "action": '{"source": 0..5}'},
            "multi_select": {"module": "multi_select_rft.py",
                             "abstain_correct": MM.REWARD_ABSTAIN_CORRECT,
                             "zero": MM.REWARD_ZERO,
                             "malformed": MM.REWARD_MALFORMED,
                             "answer_in_reward": False,
                             "action": '{"sources": [ints, ascending, 0..5개]}',
                             "reward": "2*TP/(n+k)"},
        }
    except Exception as exc:                                    # noqa: BLE001
        facts["design_from_code"]["error"] = repr(exc)

    json.dump(facts, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print("실행된 설정 %d개" % len(facts["configs"]))
    for name, row in sorted(facts["configs"].items()):
        metrics = row["metrics"]
        print("  %-14s %-14s acc %-8s n %-6s"
              % (name, row["family"],
                 ("%.4f" % metrics["accuracy"]) if "accuracy" in metrics else "—",
                 metrics.get("n", "—")))
    if facts["missing"]:
        print("\n미실행 %d건" % len(facts["missing"]))
        for row in facts["missing"]:
            print("  %-22s %s" % (row.get("config") or row.get("artifact"),
                                  row["expected"]))
    print("\n학습 입력")
    for family, row in facts["train_inputs"].items():
        print("  %-14s %-8s %s" % (family,
                                   format(row["rows"], ",") if row["rows"] else "없음",
                                   row["구성"]))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
