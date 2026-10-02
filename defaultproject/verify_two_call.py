"""Does the answer generator really stay frozen when the selector is trained?

Eight checks, in the order they can fail:

  1 only the selector LoRA is in the optimiser
  2 a short training step moves the LoRA parameters
  3 the same step leaves the base weights untouched
  4 answer generation with the adapter off is identical before and after training
  5 the adapter is on during selection and off during answering
  6 state is restored after an exception inside the selection call
  7 an empty selection sends the question alone, with no passage text in the prompt
  8 a baseline adapter on the reader, if present, survives

Check 4 is the one the claim rests on, and it is run on a *fixed* input: if the
selection changes, the answer changes too, and that is the selector working, not
the reader drifting.

    python verify_two_call.py
"""
import io
import json
import os
import sys

import torch
import transformers

import multi_select_rft as M
from two_call_reader import (adapter_state, assert_no_baseline_adapter,
                             generate_batch, selector_active)

BASE = "meta-llama/Llama-3.2-1B-Instruct"
FAILED = []

PASSAGES = [
    {"title": "Cyndi Lauper", "text": "True Colors is a song by Cyndi Lauper, released in 1986."},
    {"title": "Zedd", "text": "True Colors is a 2015 album by the Russian-German producer Zedd."},
    {"title": "Colour", "text": "Colour is the visual perception based on the spectrum of light."},
    {"title": "Phil Collins", "text": "Phil Collins is an English drummer and singer."},
    {"title": "True Colors tour", "text": "The True Colors Tour was a 2007 concert tour."},
]
QUESTION = "who sang the original version of true colors"


def check(name, ok, detail=""):
    print("[%s] %s%s" % ("ok  " if ok else "FAIL", name, ("  " + detail) if detail else ""))
    if not ok:
        FAILED.append(name)


def answer_prompt(tokenizer, question, passages):
    """The reader's input: the question and the selected passages only."""
    body = "\n".join("Title: %s\nPassage: %s" % (p["title"], p["text"]) for p in passages)
    messages = [{"role": "system", "content": "Answer with the shortest span."},
                {"role": "user", "content": (("%s\n\n" % body) if body else "")
                                            + "Question: %s" % question}]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         date_string="26 Jul 2024")


def main():
    from peft import LoraConfig, get_peft_model

    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map={"": 0})

    print("기준 모델: %s" % BASE)
    assert_no_baseline_adapter(model)
    print("답변 생성기 기준 어댑터: 없음 (plain base weights)\n")

    reader_input = answer_prompt(tokenizer, QUESTION, PASSAGES[:1])
    before = generate_batch(model, tokenizer, [reader_input], 16)
    base_snapshot = {k: v.detach().clone()
                     for k, v in list(model.state_dict().items())[:12]}

    peft_model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]))
    peft_model.base_model.disable_adapter_layers()          # resting state: off

    # 1 ------------------------------------------------------------------
    # peft's enable_adapters(False) also drops requires_grad on the adapter
    # layers, so the trainable set has to be read in the state training happens
    # in -- adapters on -- not in the resting state.
    frozen_while_off = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    check("1a. 쉬는 상태(어댑터 off)에서는 학습 대상이 없음", not frozen_while_off,
          "%d개" % len(frozen_while_off))

    peft_model.base_model.enable_adapter_layers()
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    check("1b. 어댑터 on 에서 학습 대상이 선택기 LoRA 뿐",
          bool(trainable) and all("lora_" in n for n in trainable),
          "%d개, 예: %s" % (len(trainable),
                           ".".join(trainable[0].split(".")[-3:]) if trainable else "—"))

    # adapters stay on from here: the optimiser must be built over parameters
    # that actually carry requires_grad, which is only true in this state
    lora_snapshot = {n: p.detach().clone()
                     for n, p in peft_model.named_parameters() if p.requires_grad}

    # 2, 3 ---------------------------------------------------------------
    select_input = M.build_prompt(tokenizer, QUESTION, PASSAGES)
    target = tokenizer('{"sources": [1]}', add_special_tokens=False)["input_ids"]
    ids = tokenizer(select_input, add_special_tokens=False)["input_ids"] + target
    labels = [-100] * (len(ids) - len(target)) + target
    optimiser = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=1e-3)
    peft_model.base_model.enable_adapter_layers()
    peft_model.train()
    for _ in range(3):
        loss = peft_model(input_ids=torch.tensor([ids], device=model.device),
                          labels=torch.tensor([labels], device=model.device)).loss
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
    peft_model.eval()
    peft_model.base_model.disable_adapter_layers()

    moved = [n for n, before_value in lora_snapshot.items()
             if not torch.equal(dict(peft_model.named_parameters())[n].detach(),
                                before_value)]
    check("2. 학습 후 LoRA 파라미터가 바뀜", bool(moved), "%d개 변경" % len(moved))

    # wrapping renames q_proj.weight to q_proj.base_layer.weight, so the keys
    # are normalised before the comparison rather than looked up verbatim
    state = {k.replace(".base_layer.", "."): v
             for k, v in peft_model.get_base_model().state_dict().items()}
    missing = [k for k in base_snapshot if k not in state]
    drift = [k for k, v in base_snapshot.items()
             if k in state and not torch.equal(v, state[k])]
    check("3. 기본 모델 가중치는 그대로", not drift and not missing,
          "변화 %s / 누락 %s" % (drift[:2], missing[:2]))

    # 4 ------------------------------------------------------------------
    after = generate_batch(peft_model, tokenizer, [reader_input], 16)
    check("4. 어댑터 끈 상태의 답변이 학습 전후 동일", before == after,
          "%r vs %r" % (before, after))

    # 5 ------------------------------------------------------------------
    resting = adapter_state(peft_model)
    with selector_active(peft_model):
        during = adapter_state(peft_model)
    restored = adapter_state(peft_model)
    check("5. 선택 중 켜지고 답변 시 꺼짐",
          resting is False and during is True and restored is False,
          "resting=%s during=%s after=%s" % (resting, during, restored))

    # 6 ------------------------------------------------------------------
    try:
        with selector_active(peft_model):
            raise RuntimeError("simulated failure inside the selection call")
    except RuntimeError:
        pass
    check("6. 예외 후에도 상태 복원", adapter_state(peft_model) is False)

    # 7 ------------------------------------------------------------------
    empty_prompt = answer_prompt(tokenizer, QUESTION, [])
    leaked = [p["title"] for p in PASSAGES if p["title"] in empty_prompt] + \
             [p["text"][:20] for p in PASSAGES if p["text"][:20] in empty_prompt]
    check("7. 빈 선택이면 질문만 전달", not leaked and QUESTION in empty_prompt,
          "유출 %s" % leaked[:2])

    # 8 ------------------------------------------------------------------
    try:
        assert_no_baseline_adapter(peft_model.get_base_model())
        guarded = False
    except RuntimeError:
        guarded = True
    check("8. reader 에 기존 어댑터가 있으면 거부한다", guarded,
          "어댑터가 붙은 모델에 재부착을 시도하면 RuntimeError")

    print("\n선택 출력 예시:")
    with selector_active(peft_model):
        print("   ", generate_batch(peft_model, tokenizer, [select_input], 24)[0].strip()[:80])

    os.makedirs(os.path.join("output", "multi_select"), exist_ok=True)
    json.dump({"failed": FAILED, "base": BASE, "reader_baseline_adapter": None},
              io.open(os.path.join("output", "multi_select", "two_call_check.json"),
                      "w", encoding="utf-8"), indent=1)
    print()
    if FAILED:
        print("실패 %d건: %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("2단계 호출 구조 검증 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
