"""Distil the DIG signal into GPT-small, following InfoGain-RAG's training recipe.

The paper computes DIG with a 7B model over 110K TriviaQA queries and distils it
into a RoBERTa-large cross-encoder. This runs the same algorithm at a tenth of the
scale on data that is already in the project: DIG labels from the NQ *train* split
(prepare_dig_train.py) distilled into the 63.58M GPT-small we pretrained ourselves.

    L = lambda * L_CE + (1 - lambda) * L_margin        lambda = 0.75
    CE label   DIG > b1 -> 1, DIG < b2 -> 0, in between -> excluded
    margin     log(1 + sum_neg sum_pos exp(gamma * (s_neg - s_pos)))

    python train_reranker.py --epochs 2

The scorer is a cross-encoder: "Title / Passage / Question" goes in, the classifier
reads the last position, and s = logit[1] - logit[0] is the ranking score.
"""

import argparse
import json
import math
import os
import random

import torch
import transformers
from torch.utils.data import DataLoader, Dataset

from model import TransformerForSequenceClassification

DIG_PATH = os.path.join("output", "results", "dig_train.jsonl")
OUT_DIR = os.path.join("output", "reranker")
B1, B2 = 0.5, -0.2          # paper's CE thresholds
GAMMA = 15.0                # paper's margin scale
LAMBDA = 0.75               # paper's loss balance
MAX_LEN = 320


def build_prompt(question, passage, tokenizer, max_passage_tokens=160):
    text = passage["text"]
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) > max_passage_tokens:
        text = tokenizer.decode(ids[:max_passage_tokens], clean_up_tokenization_spaces=False)
    return f"Title: {passage['title']}\nPassage: {text}\nQuestion: {question}"


class DIGGroups(Dataset):
    """One item is one query with all its labelled candidates.

    Keeping a query's candidates together is what makes the margin term possible:
    it contrasts positives against negatives *within* the same query, which is the
    comparison a reranker actually has to get right.
    """

    def __init__(self, path, tokenizer, min_pos=1, min_neg=1):
        self.tokenizer = tokenizer
        self.groups = []
        skipped = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                items = []
                for passage in record["passages"]:
                    dig = passage["dig"]
                    if dig > B1:
                        label = 1
                    elif dig < B2:
                        label = 0
                    else:
                        label = -1          # excluded from CE, still usable for margin
                    items.append((build_prompt(record["question"], passage, tokenizer),
                                  label, dig))
                pos = sum(1 for _, l, _ in items if l == 1)
                neg = sum(1 for _, l, _ in items if l == 0)
                if pos >= min_pos and neg >= min_neg:
                    self.groups.append(items)
                else:
                    skipped += 1
        print(f"[data] {len(self.groups)} usable queries, {skipped} skipped "
              f"(need >={min_pos} positive and >={min_neg} negative)")
        counts = [0, 0, 0]
        for group in self.groups:
            for _, label, _ in group:
                counts[label] += 1
        print(f"[data] candidates: {counts[1]} positive, {counts[0]} negative, "
              f"{counts[-1]} in the ignored band")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        return self.groups[index]


def collate(batch, tokenizer):
    texts, labels, owners = [], [], []
    for group_index, group in enumerate(batch):
        for text, label, _ in group:
            texts.append(text)
            labels.append(label)
            owners.append(group_index)
    tokenizer.padding_side = "left"   # the classifier reads hidden_states[:, -1, :]
    encoded = tokenizer(texts, padding="max_length", max_length=MAX_LEN,
                        truncation=True, return_tensors="pt",
                        return_token_type_ids=False)
    encoded["labels"] = torch.tensor(labels)
    encoded["owners"] = torch.tensor(owners)
    return encoded


def margin_loss(scores, labels, owners):
    """Circle-loss style contrast, computed inside each query."""
    total, groups = scores.new_zeros(()), 0
    for owner in owners.unique():
        mask = owners == owner
        pos = scores[mask & (labels == 1)]
        neg = scores[mask & (labels == 0)]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        # log(1 + sum exp(gamma * (s_neg - s_pos))) over every negative-positive pair
        diff = GAMMA * (neg.unsqueeze(0) - pos.unsqueeze(1))
        total = total + torch.logsumexp(
            torch.cat([diff.reshape(-1), diff.new_zeros(1)]), dim=0)
        groups += 1
    return total / max(groups, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", default=os.path.join("output", "pretraining", "best_model"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--queries-per-batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=402)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.init)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = DIGGroups(DIG_PATH, tokenizer)
    if not len(dataset):
        raise SystemExit("no usable queries: run prepare_dig_train.py dig first")
    loader = DataLoader(dataset, batch_size=args.queries_per_batch, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer))

    model = TransformerForSequenceClassification.from_pretrained(args.init, num_labels=2)
    model.to("cuda").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(loader) * args.epochs
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, args.warmup, steps)
    print(f"[train] {steps} steps over {args.epochs} epochs, "
          f"{args.queries_per_batch} queries per batch")

    step = 0
    for epoch in range(args.epochs):
        running = 0.0
        for batch in loader:
            labels = batch.pop("labels").to("cuda")
            owners = batch.pop("owners").to("cuda")
            batch = {k: v.to("cuda") for k, v in batch.items()}
            _, logits = model(**batch, labels=torch.zeros_like(labels).clamp(min=0))

            scores = logits[:, 1] - logits[:, 0]
            keep = labels >= 0
            ce = (torch.nn.functional.cross_entropy(logits[keep], labels[keep])
                  if keep.any() else logits.new_zeros(()))
            margin = margin_loss(scores, labels, owners)
            loss = LAMBDA * ce + (1 - LAMBDA) * margin

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            step += 1
            if step % 100 == 0:
                print(f"[train] step {step}/{steps} loss {running / 100:.4f} "
                      f"ce {ce.item():.4f} margin {margin.item():.4f}", flush=True)
                running = 0.0

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[train] saved to {args.out}")


if __name__ == "__main__":
    main()
