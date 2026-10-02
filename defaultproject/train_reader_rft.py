"""Reward-weighted LoRA fine-tuning of the reader on its own accepted rollouts.

    L = sum_j sum_{i in T_j} w_ij * l(y_ij)

`l` is the cross-entropy over the target tokens. transformers returns it already
averaged over the unmasked tokens (`fixed_cross_entropy` falls back to
reduction="mean" when `num_items_in_batch` is None), so each candidate
contributes per unit of weight regardless of how many tokens its answer took.
The prompt is masked out with -100.

The weights inside a question sum to one, which equalises how much weight each
question carries -- not how much each question moves the model. Gradient
magnitude still depends on how wrong the policy was on that sequence.

Targets are the canonical JSON of the fields the sample chose, not the raw
generation: the content is the model's own, but anything it emitted after the
closing brace is dropped rather than trained on.

    python train_reader_rft.py --rollouts output/results/rollouts_r1.jsonl \
                               --out output/reader_rft/r1
"""
import argparse
import io
import json
import os
import random
import sys
import time

import torch
import transformers

from reader_rft import build_messages, parse_output

BASE = "meta-llama/Llama-3.2-1B-Instruct"


def load_rollouts(paths):
    """Rollouts from one or more rounds.

    Later rounds accumulate rather than replace: each adapter is trained from the
    base weights on every round collected so far, so a round that happens to
    sample badly dilutes the data instead of becoming the only thing the next
    adapter sees.
    """
    groups = []
    for path in paths:
        for line in io.open(path, encoding="utf-8"):
            row = json.loads(line)
            kept = []
            for cand in row["candidates"]:
                answer, source, valid = parse_output(cand["text"])
                if not valid:
                    continue                  # should not happen; the sampler filtered
                kept.append({"target": json.dumps({"answer": answer, "source": source}),
                             "weight": cand["weight"]})
            if kept:
                groups.append((row["question"], row["passages"], kept))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="adapter to continue from")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8, help="questions per optimiser step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model

    groups = load_rollouts(args.rollouts)
    candidates = sum(len(k) for _, _, k in groups)
    print("[train] %s questions, %s candidates"
          % (format(len(groups), ","), format(candidates, ",")), flush=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.init:
        model = PeftModel.from_pretrained(model, args.init, is_trainable=True)
        print("[train] continuing from %s" % args.init, flush=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    def encode(question, passages, target):
        prompt = tokenizer.apply_chat_template(
            build_messages(question, passages), tokenize=False,
            add_generation_prompt=True, date_string="26 Jul 2024")
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [tokenizer.eos_token_id]
        ids = prompt_ids + target_ids
        return ids, [-100] * len(prompt_ids) + target_ids

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=args.lr)
    total_q = int(len(groups) * args.epochs)
    steps = max(1, total_q // args.accum)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=steps, pct_start=0.03)
    print("[train] %d optimiser steps" % steps, flush=True)

    rng = random.Random(args.seed)
    order = list(range(len(groups)))
    rng.shuffle(order)
    model.train()
    seen, step, running, started = 0, 0, 0.0, time.time()
    while seen < total_q:
        question, passages, kept = groups[order[seen % len(order)]]
        for item in kept:
            ids, labels = encode(question, passages, item["target"])
            out = model(input_ids=torch.tensor([ids], device=model.device),
                        labels=torch.tensor([labels], device=model.device))
            # w_i * l_i, spread over the questions in this accumulation window
            loss = out.loss * item["weight"] / args.accum
            loss.backward()
            running += loss.item()
        seen += 1
        if seen % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            if step < steps - 1:
                schedule.step()
            step += 1
            if step % 20 == 0:
                rate = seen / max(1e-9, time.time() - started)
                print("[train] step %d/%d  loss %.4f  %.2f q/s"
                      % (step, steps, running, rate), flush=True)
            running = 0.0

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("[train] saved adapter to %s" % args.out)


if __name__ == "__main__":
    sys.exit(main())
