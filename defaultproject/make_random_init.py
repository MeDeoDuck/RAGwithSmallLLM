"""Write a randomly initialised GPT-small checkpoint for the without-pretraining control.

The handout asks the report to compare downstream results "with and without
pretraining". Saving a random checkpoint under the same config lets the control
reuse the unmodified finetuning cells:

    python make_random_init.py
    python run_case.py classification --from-scratch
    python run_case.py rag --from-scratch
"""

import os

import transformers

from model import TransformerConfig, TransformerForCausalLM

MODEL_CONTEXT_LENGTH = 1024
ROPE_THETA = 20000.0
OUTPUT_DIR = os.path.join("output", "pretraining_random", "best_model")


def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    # identical to the 70M configuration selected in main.ipynb cell 7
    config = TransformerConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=32,
        max_postion_embeddings=MODEL_CONTEXT_LENGTH,
        attention_dropout=0.1,
        ffn_dropout=0.05,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        rope_theta=ROPE_THETA,
    )

    model = TransformerForCausalLM(config)
    total = sum(p.numel() for p in model.parameters())
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    print(f"wrote {OUTPUT_DIR} ({total / 1e6:.2f}M params, randomly initialised)")


if __name__ == "__main__":
    main()
