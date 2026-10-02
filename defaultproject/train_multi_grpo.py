"""GRPO on the multi-selection policy, using trl's implementation.

Only the reward belongs to this project. trl owns the parts that are easy to get
wrong without noticing: the behaviour policy's log-probabilities are held fixed
for the update, the ratio is formed over the same generated tokens under the
same prefix, the sampling temperature is applied the same way on both sides, no
gradient reaches the rewards or the reference model, and the loss covers the
generated tokens only. The completions are scored as produced -- nothing sorts
or re-serialises them before the reward is read, so the gradient is taken
against the tokens the policy actually emitted.

The answer generator is never called here. Selection reward needs no answer.

    python train_multi_grpo.py --out output/multi_select/grpo_r1
"""
import argparse
import io
import json
import os
import sys

import transformers

import multi_select_rft as M

BASE = "meta-llama/Llama-3.2-1B-Instruct"
TRAIN_PATH = os.path.join("output", "results", "multi_train.jsonl")

STATS = {"groups": 0, "zero_variance": 0, "malformed": 0, "scored": 0,
         "reward_sum": 0.0, "k_sum": 0, "k_counts": {}}


def make_reward_fn(flags_by_prompt):
    def selection_reward(completions, prompts=None, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            record = M.score_candidate(completion, flags_by_prompt[prompt])
            STATS["scored"] += 1
            STATS["reward_sum"] += record["reward"]
            if record["valid"]:
                STATS["k_sum"] += record["k"]
                STATS["k_counts"][record["k"]] = STATS["k_counts"].get(record["k"], 0) + 1
            else:
                STATS["malformed"] += 1
            rewards.append(record["reward"])
        STATS["groups"] += 1
        if len(set(rewards)) == 1:
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
    # trl generates per_device_train_batch_size * gradient_accumulation_steps
    # sequences in one call. With 1,150-token prompts, a reference model for the
    # KL term and 8 completions per prompt, 8*4 = 32 of them ran 16 GB dry after
    # 37 steps. Halving the accumulation halves that generation batch.
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--gen-batch", type=int, default=8,
                    help="sequences generated per call; the memory driver")
    ap.add_argument("--max-prompt-tokens", type=int, default=1400,
                    help="drop longer training prompts; 0 disables. 평가는 건드리지 않는다")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--num-shots", type=int, default=2,
                    help="must match evaluation; msel_b already ran with 2")
    args = ap.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows, flags_by_prompt, n_hist, dropped = [], {}, {}, 0
    for line in io.open(args.train, encoding="utf-8"):
        row = json.loads(line)
        prompt = M.build_prompt(tokenizer, row["question"], row["passages"],
                                num_shots=args.num_shots)
        if args.max_prompt_tokens:
            length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if length > args.max_prompt_tokens:
                dropped += 1
                continue
        flags_by_prompt[prompt] = row["flags"]
        rows.append({"prompt": prompt})
        n_hist[row["n"]] = n_hist.get(row["n"], 0) + 1
        if args.limit and len(rows) >= args.limit:
            break
    if len(flags_by_prompt) != len(rows):
        # the reward is keyed by prompt text; a collision would mislabel a group
        raise SystemExit("prompt collision: %s rows, %s distinct"
                         % (len(rows), len(flags_by_prompt)))
    print("[grpo] %s questions   n 분포 %s" % (format(len(rows), ","),
          {k: n_hist[k] for k in sorted(n_hist)}), flush=True)
    if dropped:
        # evaluation is untouched; only the training set skips these
        print("[grpo] %s개 질문을 프롬프트 길이 %d토큰 초과로 학습에서 제외"
              % (format(dropped, ","), args.max_prompt_tokens), flush=True)

    config = GRPOConfig(
        output_dir=args.out,
        num_generations=args.group,
        temperature=args.temperature,
        beta=args.beta,
        epsilon=args.epsilon,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        max_completion_length=24,
        logging_steps=10,
        # an out-of-memory crash at step 37 cost the whole run last time; with
        # checkpoints a later failure costs only the steps since the last one
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        gradient_checkpointing=True,
        bf16=True,
        # The logits tensor is batch x seq x 128,256. Measured prompts run to
        # 1,278 tokens on average and 1,404 at p99, so a generation batch of 16
        # needs 7.4 GB in bf16 and twice that once upcast -- which is what ran
        # the 16 GB card dry at step 91. This is the knob that governs it; it
        # was previously being set indirectly through gradient accumulation.
        generation_batch_size=args.gen_batch,
        report_to=[],
    )
    print("[grpo] G=%d T=%.2f beta(KL)=%.4f epsilon(clip)=%.2f lr=%.1e rank=%d"
          % (args.group, args.temperature, args.beta, args.epsilon, args.lr, args.rank),
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
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print("[grpo] 학습 대상 %s / 전체 %s (%.3f%%)"
          % (format(trainable, ","), format(total, ","), 100.0 * trainable / total),
          flush=True)

    trainer.train()
    os.makedirs(args.out, exist_ok=True)
    trainer.model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    valid = STATS["scored"] - STATS["malformed"]
    stats = dict(STATS)
    stats["zero_variance_rate"] = STATS["zero_variance"] / max(1, STATS["groups"])
    stats["used_groups"] = STATS["groups"] - STATS["zero_variance"]
    stats["malformed_rate"] = STATS["malformed"] / max(1, STATS["scored"])
    stats["mean_reward"] = STATS["reward_sum"] / max(1, STATS["scored"])
    stats["mean_k_valid"] = STATS["k_sum"] / max(1, valid)
    stats["config"] = {"group": args.group, "temperature": args.temperature,
                       "beta_kl": args.beta, "epsilon_clip": args.epsilon,
                       "lr": args.lr, "rank": args.rank,
                       "trainable_params": trainable, "total_params": total}
    json.dump(stats, io.open(os.path.join(args.out, "grpo_stats.json"), "w",
                             encoding="utf-8"), indent=1)
    print("\n[grpo] 그룹 %s개, 영분산 %s (%.1f%%), 학습에 쓰인 그룹 %s"
          % (format(STATS["groups"], ","), format(STATS["zero_variance"], ","),
             100 * stats["zero_variance_rate"], format(stats["used_groups"], ",")))
    print("[grpo] 형식 오류 %.2f%%   평균 보상 %.4f   유효 출력 평균 선택 수 %.2f"
          % (100 * stats["malformed_rate"], stats["mean_reward"], stats["mean_k_valid"]))
    print("[grpo] 선택 개수 분포 %s"
          % {k: STATS["k_counts"][k] for k in sorted(STATS["k_counts"])})
    print("[grpo] saved adapter to %s" % args.out)


if __name__ == "__main__":
    sys.exit(main())
