"""GRPO on the selector, with trl's implementation rather than a hand-rolled one.

trl's GRPOTrainer already handles the parts that are easy to get subtly wrong and
hard to notice afterwards: the behaviour policy's log-probabilities are frozen
for the update, the ratio is taken over the same generated tokens with the same
prefix, the sampling temperature is applied consistently on both sides, rewards
and the reference model carry no gradient, and the loss covers generated tokens
only. Re-deriving that here would add risk for nothing.

What this file owns is the reward: the four selection rules from
`selector_rft.py`, with the final answer deliberately absent.

    python train_selector_grpo.py --out output/selector/grpo_r1
"""
import argparse
import io
import json
import os
import sys

import torch
import transformers

import selector_rft as S

BASE = "meta-llama/Llama-3.2-1B-Instruct"
TRAIN_PATH = os.path.join("output", "results", "selector_train.jsonl")

# counts written out beside the adapter: a run whose groups are mostly degenerate
# has nothing to learn from, and that has to be visible rather than inferred
STATS = {"groups": 0, "zero_variance": 0, "malformed": 0, "scored": 0,
         "n_positive": 0, "n_zero": 0}


def make_reward_fn(flags_by_prompt):
    """trl calls this with the decoded completions for one group at a time."""

    def selection_reward(completions, prompts=None, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            flags = flags_by_prompt[prompt]
            record = S.score_candidate(completion, flags)
            STATS["scored"] += 1
            if not record["valid"]:
                STATS["malformed"] += 1
            rewards.append(record["reward"])
        values = set(rewards)
        STATS["groups"] += 1
        if len(values) == 1:
            STATS["zero_variance"] += 1
        return rewards

    return selection_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN_PATH)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    ap.add_argument("--epsilon", type=float, default=0.2, help="clip range")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows, flags_by_prompt = [], {}
    for line in io.open(args.train, encoding="utf-8"):
        row = json.loads(line)
        prompt = S.build_prompt(tokenizer, row["question"], row["passages"])
        flags_by_prompt[prompt] = row["flags"]
        rows.append({"prompt": prompt})
        STATS["n_positive" if sum(row["flags"]) > 0 else "n_zero"] += 1
        if args.limit and len(rows) >= args.limit:
            break
    print("[grpo] %s questions  (n>0 %s / n=0 %s)"
          % (format(len(rows), ","), format(STATS["n_positive"], ","),
             format(STATS["n_zero"], ",")), flush=True)
    if len(flags_by_prompt) != len(rows):
        # the reward is looked up by prompt text, so duplicates would mislabel
        raise SystemExit("prompt collision: %s rows, %s distinct prompts"
                         % (len(rows), len(flags_by_prompt)))

    config = GRPOConfig(
        output_dir=args.out,
        num_generations=args.group,
        temperature=args.temperature,
        beta=args.beta,                       # KL against the frozen reference
        epsilon=args.epsilon,                 # clip range
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        max_completion_length=12,
        max_prompt_length=None,               # the prompt is fixed-size by design
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        report_to=[],
    )
    print("[grpo] G=%d  T=%.2f  beta(KL)=%.4f  epsilon(clip)=%.2f  lr=%.1e"
          % (args.group, args.temperature, args.beta, args.epsilon, args.lr),
          flush=True)

    trainer = GRPOTrainer(
        model=BASE,
        reward_funcs=make_reward_fn(flags_by_prompt),
        args=config,
        train_dataset=Dataset.from_list(rows),
        peft_config=LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    trainer.model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    stats = dict(STATS)
    stats["zero_variance_rate"] = STATS["zero_variance"] / max(1, STATS["groups"])
    stats["malformed_rate"] = STATS["malformed"] / max(1, STATS["scored"])
    stats["used_groups"] = STATS["groups"] - STATS["zero_variance"]
    stats["config"] = {"group": args.group, "temperature": args.temperature,
                       "beta_kl": args.beta, "epsilon_clip": args.epsilon,
                       "lr": args.lr, "rank": args.rank}
    json.dump(stats, io.open(os.path.join(args.out, "grpo_stats.json"), "w",
                             encoding="utf-8"), indent=1)
    print("\n[grpo] 그룹 %s개 중 영분산 %s (%.1f%%), 학습에 쓰인 그룹 %s"
          % (format(STATS["groups"], ","), format(STATS["zero_variance"], ","),
             100 * stats["zero_variance_rate"], format(stats["used_groups"], ",")))
    print("[grpo] 형식 오류 %.2f%%  (%s / %s)"
          % (100 * stats["malformed_rate"], format(STATS["malformed"], ","),
             format(STATS["scored"], ",")))
    print("[grpo] saved adapter to %s" % args.out)


if __name__ == "__main__":
    sys.exit(main())
