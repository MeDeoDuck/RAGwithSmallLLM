"""G2: how often do G sampled answers contain a correct one, on the dev prompts?

This is a diagnostic about candidate diversity, not a performance figure. It uses
the gold answer to judge the samples, it is not a single-answer accuracy, and it
is not a ceiling on what training can reach -- training changes the distribution,
so a trained greedy decode can land either side of the untrained pass@G.

Reported split by m for the same reason the evaluation is: the questions whose
prompt never held the answer are not what the reward is about.

    python pass_at_k.py --out output/results/rft_pass8.json
"""
import argparse
import io
import json
import os
import sys
import time

import torch
import transformers

import analyze_position as A
from analyze_rft import context_flags
from reader_rft import build_prompt, is_correct, parse_output

BASE = "meta-llama/Llama-3.2-1B-Instruct"
RESULTS = os.path.join("output", "results")


def dev_questions():
    """{qid: question} -- the pool cache stores only ids, not the question text."""
    path = os.path.join(RESULTS, "prompt_rag_dig_order_output.txt")
    out = {}
    for line in io.open(path, encoding="utf-8"):
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 2:
            out[fields[0]] = fields[1]
    return out


def dev_contexts(flags):
    """The dig_order prompt for each dev question, rebuilt from the caches."""
    from analyze_rft import load_scores, rrf
    upr = load_scores("upr_scores.jsonl", "upr|")
    proxy = load_scores("proxy_dig_scores.jsonl", None)
    rows = A.load_run("dig_order")
    questions = dev_questions()
    out = []
    for line in io.open(os.path.join(RESULTS, "bm25_top100.jsonl"), encoding="utf-8"):
        row = json.loads(line)
        qid = row["qid"]
        if (qid not in flags or qid not in upr or qid not in proxy
                or qid not in rows or qid not in questions):
            continue
        pool = [p for p in row["passages"][:50] if p.get("text")]
        ids = [str(p.get("docid")) for p in pool]
        if not all(i in upr[qid] and i in proxy[qid] for i in ids):
            continue
        fused = rrf([[upr[qid][i] for i in ids], [proxy[qid][i] for i in ids]])
        top = sorted(range(len(pool)), key=lambda i: fused[i], reverse=True)[:5]
        out.append({"qid": qid, "question": questions[qid],
                    "answers": rows[qid][0], "flags": flags[qid],
                    "passages": [{"title": pool[i].get("title", ""),
                                  "text": pool[i]["text"]} for i in top]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_rows = A.load_run("dig_order")
    flags = context_flags({q: v[0] for q, v in base_rows.items()})
    items = dev_contexts(flags)
    if args.limit:
        items = items[:args.limit]
    print("[pass] %s dev questions" % format(len(items), ","), flush=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()

    stat = {"n": 0, "pass": 0, "valid": 0, "samples": 0,
            "pres_n": 0, "pres_pass": 0, "abs_n": 0, "abs_pass": 0}
    started = time.time()
    for start in range(0, len(items), args.batch):
        chunk = items[start:start + args.batch]
        prompts = [build_prompt(tokenizer, c["question"], c["passages"]) for c in chunk]
        expanded = [p for p in prompts for _ in range(args.group)]
        encoded = tokenizer(expanded, return_tensors="pt", padding=True,
                            return_token_type_ids=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**encoded, do_sample=True,
                                 temperature=args.temperature, top_p=1.0,
                                 max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id)
        replies = tokenizer.batch_decode(out[:, encoded["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
        for index, item in enumerate(chunk):
            texts = replies[index * args.group:(index + 1) * args.group]
            hit = 0
            for text in texts:
                answer, _, valid = parse_output(text)
                stat["samples"] += 1
                stat["valid"] += int(valid)
                if valid and is_correct(answer, item["answers"]):
                    hit = 1
            stat["n"] += 1
            stat["pass"] += hit
            key = "pres" if sum(item["flags"]) > 0 else "abs"
            stat[key + "_n"] += 1
            stat[key + "_pass"] += hit
        done = start + len(chunk)
        if done % (args.batch * 100) == 0 or done == len(items):
            print("[pass] %s/%s  %.2f q/s"
                  % (format(done, ","), format(len(items), ","),
                     done / max(1e-9, time.time() - started)), flush=True)

    result = {
        "group": args.group, "temperature": args.temperature, "n": stat["n"],
        "pass_at_k": stat["pass"] / max(1, stat["n"]),
        "pass_at_k_m_positive": stat["pres_pass"] / max(1, stat["pres_n"]),
        "pass_at_k_m_zero": stat["abs_pass"] / max(1, stat["abs_n"]),
        "valid_format_rate": stat["valid"] / max(1, stat["samples"]),
        "n_m_positive": stat["pres_n"], "n_m_zero": stat["abs_n"],
    }
    json.dump(result, io.open(args.out, "w", encoding="utf-8"), indent=1)
    print("\n[pass] G=%d, T=%.1f, n=%s" % (args.group, args.temperature,
                                           format(stat["n"], ",")))
    print("   pass@%d 전체   %.4f" % (args.group, result["pass_at_k"]))
    print("   pass@%d m>0   %.4f  (n=%s)" % (args.group, result["pass_at_k_m_positive"],
                                             format(stat["pres_n"], ",")))
    print("   pass@%d m=0   %.4f  (n=%s)" % (args.group, result["pass_at_k_m_zero"],
                                             format(stat["abs_n"], ",")))
    print("   형식 유효율    %.4f" % result["valid_format_rate"])
    print("[pass] wrote %s" % args.out)


if __name__ == "__main__":
    sys.exit(main())
