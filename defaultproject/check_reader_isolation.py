"""Does the selector's adapter leak into the reader?

The experiment's claim is that the answer model was not trained, so this has to
be checked rather than assumed. Two tests:

  1. the reader's module tree holds no PEFT layers
  2. the reader produces byte-identical generations with and without a selector
     adapter loaded in the same process

Test 2 is the one that matters. A shared-base implementation that toggles the
adapter can pass test 1 and still read with the adapter on if a context manager
is missed somewhere.

    python check_reader_isolation.py --adapter output/selector/grpo_r1
"""
import argparse
import sys

import torch
import transformers

FAILED = []


def check(name, ok, detail=""):
    print("[%s] %s%s" % ("ok  " if ok else "FAIL", name,
                         ("  " + detail) if detail else ""))
    if not ok:
        FAILED.append(name)


def generate(model, tokenizer, prompts, max_new_tokens=24):
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        return_token_type_ids=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id)
    tokenizer.padding_side = side
    return tokenizer.batch_decode(out[:, encoded["input_ids"].shape[1]:],
                                  skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default="meta-llama/Llama-3.2-1B-Instruct")
    args = ap.parse_args()

    from peft import PeftModel

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "Title: Cyndi Lauper\nPassage: True Colors is a song by Cyndi Lauper.\n\n"
        "Question: who sang true colors\nAnswer:",
        "Title: Saturn\nPassage: Titan is the largest moon of Saturn.\n\n"
        "Question: what is the largest moon of saturn\nAnswer:",
    ]

    reader = transformers.AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    before = generate(reader, tokenizer, prompts)
    state_before = {k: v.clone() for k, v in list(reader.state_dict().items())[:8]}

    # the selector lives on its own copy; loading it must not reach the reader
    selector_base = transformers.AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map={"": 0})
    selector = PeftModel.from_pretrained(selector_base, args.adapter).eval()

    check("reader 에 PEFT 레이어 없음",
          not any(type(m).__name__.startswith("Lora") for m in reader.modules()))
    check("selector 에는 PEFT 레이어 있음",
          any(type(m).__name__.startswith("Lora") for m in selector.modules()))
    check("두 모델이 서로 다른 객체", reader is not selector_base)

    after = generate(reader, tokenizer, prompts)
    check("adapter 적재 후에도 reader 출력 동일",
          before == after, "%r vs %r" % (before, after))

    drift = [k for k, v in state_before.items()
             if not torch.equal(v, reader.state_dict()[k])]
    check("reader 가중치 변화 없음", not drift, str(drift[:3]))

    # the selector must actually differ, otherwise test 4 passes vacuously
    sel_out = generate(selector, tokenizer, prompts)
    print("     reader   : %r" % before)
    print("     selector : %r" % sel_out)
    if sel_out == before:
        print("[warn] selector 출력이 base 와 같다 — adapter 효과가 없거나 "
              "학습이 안 된 어댑터일 수 있다 (격리 실패는 아님)")

    print()
    if FAILED:
        print("실패 %d건: %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("reader 격리 확인됨")
    return 0


if __name__ == "__main__":
    sys.exit(main())
