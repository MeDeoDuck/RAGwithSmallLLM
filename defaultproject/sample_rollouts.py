"""Collect G rollouts per training question, score them, and keep the usable ones.

Everything that decides whether a sample trains the model is recorded, not just
the samples that survive: with one- and two-word answers a group can sample the
same string eight times, and a run whose groups are mostly degenerate has nothing
to learn from however long it is left going. The skip tally is the measurement
that says whether the next stage is worth starting.

    python sample_rollouts.py --out output/results/rollouts_r1.jsonl
    python sample_rollouts.py --adapter output/reader_rft/r1 --out .../rollouts_r2.jsonl
"""
import argparse
import io
import json
import os
import sys
import time

import torch
import transformers

from reader_rft import build_prompt, group_advantage, score_candidate

BASE = "meta-llama/Llama-3.2-1B-Instruct"
TRAIN_PATH = os.path.join("output", "results", "reader_rft_train.jsonl")


def load_questions(path, limit=None):
    rows = []
    for line in io.open(path, encoding="utf-8"):
        row = json.loads(line)
        if row["m"] <= 0:                      # m = 0 is not a training target
            continue
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA adapter; omit for the base policy")
    ap.add_argument("--train", default=TRAIN_PATH)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4, help="questions per forward batch")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_questions(args.train, args.limit or None)
    print("[roll] %s questions (m>0)" % format(len(rows), ","), flush=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map={"": 0})
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print("[roll] adapter %s" % args.adapter, flush=True)
    model.eval()

    skip = {"no_valid": 0, "zero_variance": 0, "no_correct_candidate": 0}
    kept_q = kept_c = 0
    valid_total = sampled_total = 0
    started = time.time()
    handle = io.open(args.out, "w", encoding="utf-8")

    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        prompts = [build_prompt(tokenizer, r["question"], r["passages"]) for r in chunk]
        # every rollout of a question shares its prompt, so they are generated
        # together and the prefill is paid once per question per batch
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

        for index, row in enumerate(chunk):
            texts = replies[index * args.group:(index + 1) * args.group]
            records = [score_candidate(t, row["answers"], row["flags"]) for t in texts]
            sampled_total += len(records)
            valid_total += sum(1 for r in records if r["valid"])
            weights, reason = group_advantage(records)
            if reason is not None:
                skip[reason] += 1
                continue
            kept_q += 1
            kept_c += len(weights)
            handle.write(json.dumps({
                "qid": row["qid"], "question": row["question"],
                "answers": row["answers"], "passages": row["passages"],
                "flags": row["flags"], "m": row["m"],
                "candidates": [{"text": records[i]["text"], "weight": w,
                                "reward": records[i]["reward"],
                                "source": records[i]["source"], "g": records[i]["g"]}
                               for i, w in sorted(weights.items())],
            }, ensure_ascii=False) + "\n")

        done = start + len(chunk)
        if done % (args.batch * 25) == 0 or done == len(rows):
            rate = done / max(1e-9, time.time() - started)
            print("[roll] %s/%s  kept %s  %.2f q/s"
                  % (format(done, ","), format(len(rows), ","),
                     format(kept_q, ","), rate), flush=True)
    handle.close()

    total = len(rows)
    print("\n[roll] 샘플 %s개 중 형식 유효 %s (%.1f%%)"
          % (format(sampled_total, ","), format(valid_total, ","),
             100.0 * valid_total / max(1, sampled_total)))
    print("[roll] 질문 %s개의 처리 결과" % format(total, ","))
    print("   %-22s %7s %7s" % ("", "질문", "비율"))
    print("   %-22s %7s %6.1f%%" % ("학습에 사용", format(kept_q, ","),
                                    100.0 * kept_q / max(1, total)))
    for reason, label in (("no_valid", "형식 전멸"),
                          ("zero_variance", "영분산"),
                          ("no_correct_candidate", "정답 후보 없음")):
        print("   %-22s %7s %6.1f%%" % (label, format(skip[reason], ","),
                                        100.0 * skip[reason] / max(1, total)))
    print("[roll] 유지된 후보 %s개 (질문당 평균 %.2f)"
          % (format(kept_c, ","), kept_c / max(1, kept_q)))
    print("[roll] wrote %s" % args.out)


if __name__ == "__main__":
    sys.exit(main())
