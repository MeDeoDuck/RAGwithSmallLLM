"""Selection and answering as two calls on one shared base model.

The adapter is kept **off** as the resting state. Selection turns it on inside a
context manager and turns it off again in `finally`; everything else -- including
the answer generation the evaluation harness performs on `rag.model` -- therefore
runs on the frozen base weights without needing to remember anything.

Doing it the other way round (adapter on by default, disabled around the answer
call) would put the burden on every future caller to wrap correctly, and the
answer call does not happen inside this file: the harness generates from
`rag.model` after `search()` has returned. Resting-off is the only arrangement
that survives that.

Verified against peft 0.21.1: `PeftModel.base_model.enable_adapter_layers()` and
`.disable_adapter_layers()` are the methods `PeftModel.disable_adapter()` itself
uses, and `PeftModel.get_model_status().enabled` reports the current state.

Limitation, stated rather than assumed: `enable_adapter_layers()` toggles *all*
adapters on the model. That is correct here only because the answer generator
carries no adapter of its own -- `assert_no_baseline_adapter` checks that and
refuses otherwise. A reader that did carry one would need named adapters and
`set_adapter`, which is not what this file implements.
"""
import contextlib

import torch


def adapter_layer_names(model):
    """Names of any PEFT tuner layers in the module tree.

    Matching on the class name does not work: peft's LoRA linear layer is
    `peft.tuners.lora.layer.Linear`, so `type(m).__name__` is just "Linear" and
    a name-based guard silently finds nothing. `BaseTunerLayer` is the type all
    of them share.
    """
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:                                   # peft not installed
        return [name for name, module in model.named_modules()
                if hasattr(module, "lora_A")]
    return [name for name, module in model.named_modules()
            if isinstance(module, BaseTunerLayer)]


def assert_no_baseline_adapter(model):
    """The reader must be plain base weights for the resting-off scheme to hold."""
    found = adapter_layer_names(model)
    if found:
        raise RuntimeError(
            "answer generator already carries %d adapter layer(s) (e.g. %s); this "
            "module assumes an adapter-free reader and would disable the reader's "
            "own adapter together with the selector's" % (len(found), found[:2]))


def adapter_state(peft_model):
    """Current enabled state as peft reports it: True / False / 'irregular'."""
    return peft_model.get_model_status().enabled


@contextlib.contextmanager
def selector_active(peft_model):
    """Turn the selector adapter on for the duration, then restore the baseline.

    The baseline is read rather than assumed, and restored in `finally`, so an
    exception inside the selection call cannot leave the model in a state where
    the next answer call silently runs on trained weights.
    """
    baseline = adapter_state(peft_model)
    peft_model.base_model.enable_adapter_layers()
    try:
        yield peft_model
    finally:
        if baseline is True:
            peft_model.base_model.enable_adapter_layers()
        else:
            peft_model.base_model.disable_adapter_layers()


def attach_selector(model, adapter_path):
    """Wrap the reader in a PeftModel whose adapter rests disabled.

    Returns the PeftModel. `model` is the same underlying module, so the harness
    keeps generating from the frozen weights unless `selector_active` is open.
    """
    from peft import PeftModel

    assert_no_baseline_adapter(model)
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    peft_model.base_model.disable_adapter_layers()      # resting state: off
    for parameter in peft_model.parameters():
        parameter.requires_grad_(False)                 # nothing is trained here
    return peft_model.eval()


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens, batch_size=8):
    """Greedy generation, left padding, prompt tokens stripped from the output."""
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    out = []
    try:
        for start in range(0, len(prompts), max(1, batch_size)):
            chunk = prompts[start:start + max(1, batch_size)]
            encoded = tokenizer(chunk, return_tensors="pt", padding=True,
                                return_token_type_ids=False).to(model.device)
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens,
                                       do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
            out += tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
    finally:
        tokenizer.padding_side = side
    return out
