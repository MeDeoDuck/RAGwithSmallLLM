"""Build the NQ train HF dataset without loading the 7.4 GB JSON into memory.

dataset/rag.py converts the DPR file with pandas.read_json followed by
Dataset.from_pandas, which needs well over 20 GB of RAM and gets OOM-killed on a
smaller machine. This script streams the JSON array instead and writes the same
dataset to the same path, so donwload_dataset_nq_open_dpr() finds it and skips
the conversion. No provided file is modified.

Only the fields RAGDataset actually reads are kept. The handout states that the
negative samples are for training the DPR retriever and are not used here, and
RAGDataset.__getitem__ never touches them.

    python prepare_nq_train.py
"""

import json
import os
import shutil

from datasets import load_dataset

ROOT = os.path.join("local_cache", "rag", "data", "nq_open_dpr")
SRC = os.path.join(ROOT, "nq_train.json")
JSONL = os.path.join(ROOT, "nq_train.stream.jsonl")
OUT = os.path.join(ROOT, "nq_train")

CTX_FIELDS = ("title", "text", "passage_id")


def stream_json_array(path, chunk_size=1 << 22):
    """Yield objects from a top-level JSON array without holding the whole file."""
    decoder = json.JSONDecoder()
    buf = ""
    with open(path, encoding="utf-8") as handle:
        while True:  # skip to the opening bracket
            char = handle.read(1)
            if not char:
                return
            if char == "[":
                break
        while True:
            buf = buf.lstrip()
            if buf.startswith(","):
                buf = buf[1:].lstrip()
            if buf.startswith("]"):
                return
            try:
                obj, end = decoder.raw_decode(buf)
            except ValueError:
                more = handle.read(chunk_size)
                if not more:
                    return
                buf += more
                continue
            yield obj
            buf = buf[end:]


def main():
    if os.path.isdir(OUT):
        print(f"{OUT} already exists, nothing to do")
        return
    if not os.path.exists(SRC):
        raise SystemExit(f"{SRC} not found (let the notebook download it first)")

    size_gb = os.path.getsize(SRC) / 2**30
    print(f"streaming {SRC} ({size_gb:.2f} GB)")

    kept = 0
    with open(JSONL, "w", encoding="utf-8") as out:
        for index, record in enumerate(stream_json_array(SRC)):
            contexts = [
                {field: str(ctx.get(field, "")) for field in CTX_FIELDS}
                for ctx in record.get("positive_ctxs", [])
            ]
            row = {
                "uid": f"train_{index:06d}",
                "question": record.get("question", ""),
                "answers": [str(a) for a in record.get("answers", [])],
                "positive_ctxs": contexts,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 5000 == 0:
                print(f"  {kept} rows", flush=True)

    print(f"wrote {kept} rows to {JSONL} ({os.path.getsize(JSONL) / 2**30:.2f} GB)")

    dataset = load_dataset("json", data_files=JSONL, split="train")
    print(f"loaded: {dataset}")
    dataset.save_to_disk(OUT)
    print(f"saved to {OUT}")

    os.remove(JSONL)
    os.remove(SRC)
    print("removed the intermediate jsonl and the source json")
    print(f"free space check: {shutil.disk_usage('.').free / 2**30:.0f} GB")


if __name__ == "__main__":
    main()
