"""Training inputs that are the pipeline's own output, not gold-shaped contexts.

The earlier reader experiment built its five passages by sampling from the pool
to hit a target distribution of answer-bearing counts. That is not allowed here:
the selector has to meet exactly the inputs it will meet at evaluation, so the
five come from running the real chain on the train split.

    BM25 top-20  ->  UPR  ->  first-pass answer  ->  ALR  ->  RRF  ->  top-5  ->  edge_last

The chain is executed by `ImprovedRAG` itself rather than reimplemented, so the
ranking logic cannot drift from the evaluation path. The one difference from dev
is depth: train pools were retrieved at top-20, dev at top-50. That is recorded,
not hidden.

Nothing here reconstructs a context from gold. Gold answers are read only to
compute the labels the reward will use, after the five are fixed.

    python prepare_multi_train.py --questions 6000
"""
import argparse
import ast
import io
import json
import os
import sys
import time

import torch
import transformers

from improved_rag import ImprovedRAG
from multi_select_rft import passage_flags

BASE = "meta-llama/Llama-3.2-1B-Instruct"
RESULTS = os.path.join("output", "results")
TRAIN_POOL = os.path.join(RESULTS, "bm25_train.jsonl")
FIRST_PASS = os.path.join(RESULTS, "multi_train_firstpass.txt")
UPR_CACHE = os.path.join(RESULTS, "multi_train_upr.jsonl")
PROXY_CACHE = os.path.join(RESULTS, "multi_train_proxy.jsonl")
OUT_PATH = os.path.join(RESULTS, "multi_train.jsonl")


def load_pool(limit):
    rows = []
    for line in io.open(TRAIN_POOL, encoding="utf-8"):
        row = json.loads(line)
        if len([p for p in row["passages"] if p.get("text")]) < 5:
            continue
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def attach(rag, model, tokenizer):
    rag.set_model(model)
    rag.set_tokenizer(tokenizer)
    return rag


def first_pass(rows, model, tokenizer, batch):
    """Greedy v3 answers over the UPR-reranked top-5, as ALR expects them.

    On dev the first-pass file is `prompt_rag_slot_edge_last_output.txt` -- the
    UPR-reranked, slot-reordered run, not raw BM25. Matching that here keeps the
    pseudo-answer distribution comparable, and it also avoids reading `score`
    off the train pool: `bm25_train.jsonl` carries only docid/title/text, so the
    no-rerank path in `ranked_pool` has nothing to sort by. The UPR scores land
    in UPR_CACHE and the second phase reuses them, so the pass is not wasted.
    """
    if os.path.exists(FIRST_PASS) and os.path.getsize(FIRST_PASS):
        print("[multi] first pass cached: %s" % FIRST_PASS, flush=True)
        return
    rag = attach(ImprovedRAG(variant="v3", k_pool=20, cache_path=TRAIN_POOL,
                             rerank_cache_path=UPR_CACHE, slot_order="edge_last"),
                 model, tokenizer)
    handle = io.open(FIRST_PASS, "w", encoding="utf-8")
    started = time.time()
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        questions = [r["question"] for r in chunk]
        qids = [r["qid"] for r in chunk]
        prompts = rag.make_augmented_inputs_for_generate(questions, qids, k=5)
        side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                            return_token_type_ids=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**encoded, max_new_tokens=rag.max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        tokenizer.padding_side = side
        raw = tokenizer.batch_decode(out[:, encoded["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        parsed = rag.postprocess(raw)
        for row, text, answer in zip(chunk, raw, parsed):
            handle.write("\t".join([row["qid"], row["question"], repr(row["answers"]),
                                    answer.replace("\t", " "),
                                    text.replace("\t", " ").replace("\n", " ")]) + "\n")
        done = start + len(chunk)
        if done % (batch * 25) == 0 or done == len(rows):
            print("[multi] first pass %s/%s  %.2f q/s"
                  % (format(done, ","), format(len(rows), ","),
                     done / max(1e-9, time.time() - started)), flush=True)
    handle.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    if "train" not in os.path.basename(TRAIN_POOL):
        raise SystemExit("DEV_GUARD: %s is not a train pool" % TRAIN_POOL)

    rows = load_pool(args.questions)
    print("[multi] %s train questions" % format(len(rows), ","), flush=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()

    first_pass(rows, model, tokenizer, args.batch)

    # the evaluation chain, pointed at the train pool and train caches
    rag = attach(ImprovedRAG(variant="v3", k_pool=20, cache_path=TRAIN_POOL,
                             rerank_cache_path=UPR_CACHE, use_proxy=True, fuse=True,
                             pseudo_path=FIRST_PASS, proxy_cache_path=PROXY_CACHE,
                             slot_order="edge_last"), model, tokenizer)

    written, n_hist, started = 0, {}, time.time()
    handle = io.open(args.out, "w", encoding="utf-8")
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        questions = [r["question"] for r in chunk]
        qids = [r["qid"] for r in chunk]
        passages, _ = rag.search(questions, qids, k=5)
        for row, context in zip(chunk, passages):
            flags = passage_flags(context, row["answers"])
            handle.write(json.dumps({
                "qid": row["qid"], "question": row["question"],
                "answers": row["answers"],
                "passages": [{"id": p.get("id"), "title": p.get("title", ""),
                              "text": p.get("text", "")} for p in context],
                "flags": flags, "n": sum(flags),
                "truncated": rag.ctx_token_budget is not None,
            }, ensure_ascii=False) + "\n")
            written += 1
            n_hist[sum(flags)] = n_hist.get(sum(flags), 0) + 1
        done = start + len(chunk)
        if done % (args.batch * 25) == 0 or done == len(rows):
            print("[multi] pipeline %s/%s  %.2f q/s"
                  % (format(done, ","), format(len(rows), ","),
                     done / max(1e-9, time.time() - started)), flush=True)
    handle.close()

    print("\n[multi] wrote %s questions to %s" % (format(written, ","), args.out))
    print("[multi] n 분포 (실제 파이프라인 출력, 맞추지 않음)")
    dev = {0: 39.3, 1: 18.5, 2: 14.0, 3: 10.0, 4: 7.9, 5: 10.5}
    print("   %3s %9s %9s %14s" % ("n", "문항", "train", "dev dig_order"))
    for n in range(6):
        got = n_hist.get(n, 0)
        print("   %3d %9s %8.1f%% %13.1f%%"
              % (n, format(got, ","), 100.0 * got / max(1, written), dev[n]))
    print("\n[multi] 차이: train pool 깊이 20 vs dev 50. 분포를 맞추지 않았다 — "
          "실제 파이프라인 출력을 그대로 쓰기로 했기 때문이다.")


if __name__ == "__main__":
    sys.exit(main())
