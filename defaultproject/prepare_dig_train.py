"""Build InfoGain-RAG's DIG training set from the NQ *train* split.

This is the paper's data-collection step (its Algorithm 1) at one tenth the scale:
for a sample of training questions, retrieve BM25 candidates and label each one by
how much it raises the likelihood of the gold answer.

    DIG(d | x) = p(y | x, d) - p(y | x)

The gold answer is a *training* label here, so nothing about the dev set is touched.

    python prepare_dig_train.py bm25 --num-questions 10000 --k 20   # CPU
    python prepare_dig_train.py dig                                  # GPU
"""

import argparse
import json
import os

import torch

DEV_GUARD = "nq_dev"
TRAIN_PATH = os.path.join("local_cache", "rag", "data", "nq_open_dpr", "nq_train")
INDEX_PATH = os.path.join("local_cache", "rag",
                          "lucene-index.wikipedia-dpr-100w.20210120.d1b9e6")
POOL_PATH = os.path.join("output", "results", "bm25_train.jsonl")
DIG_PATH = os.path.join("output", "results", "dig_train.jsonl")
SCORER = "meta-llama/Llama-3.2-1B-Instruct"
MAX_PASSAGE_TOKENS = 160


def load_train(num_questions):
    from datasets import Dataset as HF_Dataset

    if DEV_GUARD in TRAIN_PATH:
        raise SystemExit("refusing to build training labels from the dev split")
    dataset = HF_Dataset.load_from_disk(TRAIN_PATH)
    if num_questions and num_questions < len(dataset):
        dataset = dataset.select(range(num_questions))
    return dataset


def build_pool(num_questions, k, batch_size=64):
    from pyserini.search.lucene import LuceneSearcher
    from utils.etc import hit2docdict

    dataset = load_train(num_questions)
    searcher = LuceneSearcher(index_dir=INDEX_PATH)
    os.makedirs(os.path.dirname(POOL_PATH), exist_ok=True)
    questions, answers = dataset["question"], dataset["answers"]

    with open(POOL_PATH, "w", encoding="utf-8") as out:
        for start in range(0, len(questions), batch_size):
            chunk_q = list(questions[start:start + batch_size])
            qids = [f"train_{i:06d}" for i in range(start, start + len(chunk_q))]
            hits = searcher.batch_search(queries=chunk_q, qids=qids, k=k, threads=8)
            for offset, qid in enumerate(qids):
                passages = []
                for hit in hits[qid]:
                    doc = hit2docdict(hit)
                    title, sep, text = doc["contents"].partition("\n")
                    if not sep:
                        title, text = "", title
                    passages.append({"docid": str(hit.docid),
                                     "title": title.strip().strip('"'),
                                     "text": text.strip()})
                out.write(json.dumps({"qid": qid,
                                      "question": chunk_q[offset],
                                      "answers": list(answers[start + offset]),
                                      "passages": passages}) + "\n")
            print(f"[bm25] {min(start + batch_size, len(questions))}/{len(questions)}",
                  flush=True)
    print(f"[bm25] wrote {POOL_PATH}")


class Scorer:
    """Teacher-forced mean log-likelihood of a target under a prefix."""

    def __init__(self, name=SCORER):
        import transformers

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True)
        self.model.eval()

    def prompt(self, question, passage):
        text = passage["text"]
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_PASSAGE_TOKENS:
            text = self.tokenizer.decode(ids[:MAX_PASSAGE_TOKENS],
                                         clean_up_tokenization_spaces=False)
        return f"Title: {passage['title']}\nPassage: {text}\nQuestion: {question}\nAnswer:"

    @torch.no_grad()
    def score(self, prefixes, target, prefix_len=256):
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        if not target_ids or not prefixes:
            return [0.0] * len(prefixes)
        t_len = len(target_ids)
        device = self.model.device
        target_tensor = torch.tensor(target_ids, device=device).unsqueeze(0)
        enc = self.tokenizer(prefixes, padding="max_length", max_length=prefix_len,
                             truncation=True, return_tensors="pt",
                             return_token_type_ids=False)
        batch = enc["input_ids"].size(0)
        input_ids = torch.cat([enc["input_ids"].to(device),
                               target_tensor.expand(batch, t_len)], dim=1)
        mask = torch.cat([enc["attention_mask"].to(device),
                          torch.ones((batch, t_len), dtype=enc["attention_mask"].dtype,
                                     device=device)], dim=1)
        out = self.model(input_ids=input_ids, attention_mask=mask, logits_to_keep=t_len + 1)
        logits = out.logits[:, -(t_len + 1):-1, :]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        picked = logprobs.gather(-1, input_ids[:, -t_len:].unsqueeze(-1)).squeeze(-1)
        return (picked.sum(-1) / t_len).tolist()


def label_dig(batch_size=40):
    if not os.path.exists(POOL_PATH):
        raise SystemExit(f"{POOL_PATH} not found: run `prepare_dig_train.py bm25` first")
    scorer = Scorer()
    done = set()
    if os.path.exists(DIG_PATH):
        with open(DIG_PATH, encoding="utf-8") as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)["qid"])
                except ValueError:
                    continue
        print(f"[dig] resuming, {len(done)} questions already labelled", flush=True)

    written = 0
    with open(POOL_PATH, encoding="utf-8") as src, \
            open(DIG_PATH, "a", encoding="utf-8") as out:
        for line in src:
            record = json.loads(line)
            if record["qid"] in done:
                continue
            answers = [a.strip() for a in record["answers"] if a.strip()]
            if not answers:
                continue
            target = " " + answers[0]
            question = record["question"]
            passages = record["passages"]

            scores = []
            for start in range(0, len(passages), batch_size):
                chunk = passages[start:start + batch_size]
                scores.extend(scorer.score(
                    [scorer.prompt(question, p) for p in chunk], target))
            baseline = scorer.score([f"Question: {question}\nAnswer:"], target)[0]

            out.write(json.dumps({
                "qid": record["qid"], "question": question, "answers": answers,
                "baseline": baseline,
                "passages": [{"docid": p["docid"], "title": p["title"], "text": p["text"],
                              "dig": s - baseline} for p, s in zip(passages, scores)],
            }) + "\n")
            written += 1
            if written % 200 == 0:
                out.flush()
                print(f"[dig] {written} questions labelled", flush=True)
    print(f"[dig] wrote {written} questions to {DIG_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bm25", help="retrieve candidates for training questions (CPU)")
    b.add_argument("--num-questions", type=int, default=10000)
    b.add_argument("--k", type=int, default=20)
    d = sub.add_parser("dig", help="label each candidate with its information gain (GPU)")
    d.add_argument("--batch-size", type=int, default=40)
    args = parser.parse_args()

    if args.command == "bm25":
        build_pool(args.num_questions, args.k)
    else:
        label_dig(args.batch_size)


if __name__ == "__main__":
    main()
