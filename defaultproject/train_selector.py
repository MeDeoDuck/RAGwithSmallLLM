"""LoRA warm start for the generative selector, on the NQ *train* split only.

The reward the selector is eventually optimised for -- how many of its five picks
contain the gold answer -- has a known argmax on the train split, so this stage is
plain supervised learning on that argmax. It exists to get the policy producing
well-formed, non-random selections; the reward's ties (a question with eight
answer-bearing candidates has 56 optimal subsets) are what the later RL stage is
for, and this stage deliberately teaches only one of them.

    python train_selector.py --epochs 1 --batch 1 --accum 8
"""
import argparse
import io
import json
import os
import random
import sys

import torch
import transformers

from select_rag import build_prompt, format_target

TRAIN_PATH = "output/results/select_train.jsonl"
POOL_PATH = "output/results/bm25_train.jsonl"
OUT_DIR = os.path.join("output", "selector")
BASE = "meta-llama/Llama-3.2-1B-Instruct"


def load_examples(limit=None):
    pools = {}
    for line in io.open(POOL_PATH, encoding="utf-8"):
        row = json.loads(line)
        pools[row["qid"]] = [p for p in row["passages"] if p.get("text")]
    examples = []
    for line in io.open(TRAIN_PATH, encoding="utf-8"):
        row = json.loads(line)
        pool = pools.get(row["qid"])
        if not pool:
            continue
        examples.append((row["question"], pool, row["target"]))
        if limit and len(examples) >= limit:
            break
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    examples = load_examples(args.limit or None)
    print("[sel] %s training questions" % format(len(examples), ","), flush=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    def encode(question, pool, target):
        prompt = build_prompt(question, pool, tokenizer)
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(format_target(target) + tokenizer.eos_token,
                               add_special_tokens=False)["input_ids"]
        ids = prompt_ids + target_ids
        # the prompt is context, not something to predict
        labels = [-100] * len(prompt_ids) + target_ids
        return ids, labels

    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total = int(len(examples) * args.epochs)
    steps = max(1, total // (args.batch * args.accum))
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=steps, pct_start=0.03)
    print("[sel] %d optimiser steps" % steps, flush=True)

    random.seed(0)
    order = list(range(len(examples)))
    random.shuffle(order)
    model.train()
    seen, step, running = 0, 0, 0.0
    while seen < total:
        index = order[seen % len(order)]
        ids, labels = encode(*examples[index])
        batch = {"input_ids": torch.tensor([ids], device=model.device),
                 "labels": torch.tensor([labels], device=model.device)}
        loss = model(**batch).loss / args.accum
        loss.backward()
        running += loss.item()
        seen += 1
        if seen % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            if step < steps - 1:
                schedule.step()
            step += 1
            if step % 25 == 0:
                print("[sel] step %d/%d  loss %.4f" % (step, steps, running), flush=True)
            running = 0.0

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("[sel] saved adapter to %s" % args.out)


if __name__ == "__main__":
    sys.exit(main())
