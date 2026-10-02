"""Baseline RAG for handout section 4.2.1.

Prompting variants on top of ModelRAG for Llama-3.2-1B-Instruct, plus the
answer-parsing chain used to post-process generations before scoring with
utils.metrics.best_subspan_exact_match.
"""

import json

import torch

from model_rag import ModelRAG
from rag_parsing import PARSERS

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYS_CORE = """You are a question answering system. You are given Wikipedia passages and one question.

Rules:
- Use the passages to answer. The answer is a span of text, usually a name, a date, a number, or a short title.
- The answer is at most 5 words.
- Do not write a sentence. Do not repeat the question. Do not explain your choice.
- If the passages do not contain the answer, give your single best guess in the same short form. Never reply "unknown", "not stated", "not mentioned", or "I don't know"."""

SYS_TERSE = SYS_CORE + """
- Output format: the answer only, on a single line, with no label and no punctuation at the end."""

SYS_JSON = SYS_CORE + """
- Output format: one line of JSON and nothing else, in exactly this form:
{"answer": "<the answer>"}
Do not wrap the JSON in a code block. Do not add any other key. Do not add any text before or after the JSON."""

SYS_COT = SYS_CORE + """
- Output format: exactly two lines and nothing else.
Line 1 must start with "Reasoning: " followed by one sentence of at most 25 words naming the passage that contains the answer.
Line 2 must start with "Answer: " followed by the answer only, at most 5 words.
Never write anything after line 2."""

USER_TEMPLATE = """{passages}

Question: {question}"""

EX1_USER = """Title: Marie Curie
Passage: Marie Curie was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity. In 1903 she became the first woman to win a Nobel Prize, sharing the Nobel Prize in Physics with her husband Pierre Curie and Henri Becquerel.

Question: who was the first woman to win a nobel prize"""
EX1_GOLD = "Marie Curie"
EX1_COT = ("Reasoning: The Marie Curie passage states that in 1903 she became the first woman to win a Nobel Prize.\n"
           "Answer: Marie Curie")

EX2_USER = """Title: Eiffel Tower
Passage: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889 as the centrepiece of the 1889 World's Fair, and was the tallest structure in the world until 1930.

Question: when was the eiffel tower finished being built"""
EX2_GOLD = "1889"
EX2_COT = ("Reasoning: The Eiffel Tower passage says construction ran from 1887 to 1889 for the World's Fair.\n"
           "Answer: 1889")

EX3_USER = """Title: Titan (moon)
Passage: Titan is the largest moon of Saturn and the second-largest natural satellite in the Solar System. It is the only moon known to have a dense atmosphere, and the only known body in space other than Earth with stable bodies of surface liquid.
Title: Saturn
Passage: Saturn is the sixth planet from the Sun and the second-largest in the Solar System, after Jupiter. It is a gas giant with an average radius about nine and a half times that of Earth.

Question: what is the largest moon of saturn"""
EX3_GOLD = "Titan"
EX3_COT = ("Reasoning: The Titan passage says Titan is the largest moon of Saturn.\n"
           "Answer: Titan")

EX4_USER = """Title: Blue whale
Passage: The blue whale is a marine mammal and the largest animal known to have ever existed. Reaching a maximum confirmed length of 29.9 metres and a weight of 199 tonnes, it is found in every ocean except the Arctic.

Question: what is the largest animal that has ever lived"""
EX4_GOLD = "blue whale"
EX4_COT = ("Reasoning: The blue whale passage says it is the largest animal known to have ever existed.\n"
           "Answer: blue whale")

# order matters: index 0..3 are taken for num_shots = 1..4
EXEMPLARS = [
    (EX1_USER, EX1_GOLD, EX1_COT),
    (EX2_USER, EX2_GOLD, EX2_COT),
    (EX3_USER, EX3_GOLD, EX3_COT),
    (EX4_USER, EX4_GOLD, EX4_COT),
]

# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

VARIANTS = {
    # name:      (system prompt, num_shots, parser, max_new_tokens, use_chat_template)
    "v0":        (None,      0, "plain", 16, False),
    "v1":        (SYS_TERSE, 0, "plain", 16, True),
    "v2":        (SYS_TERSE, 2, "plain", 16, True),
    "v3":        (SYS_JSON,  2, "json",  32, True),
    "v4":        (SYS_COT,   1, "cot",   96, True),
}


class PromptedRAG(ModelRAG):
    """ModelRAG with Llama chat formatting, few-shot exemplars and answer parsing.

    variant "v0" keeps the naive plain-completion prompt and only adopts the
    controlled decoding settings, so it isolates the effect of prompting.
    """

    def __init__(self, variant="v2", num_shots=None, max_new_tokens=None,
                 date_string="26 Jul 2024", ctx_token_budget=None,
                 pad_to_multiple_of=64):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        system, shots, parser, tokens, use_chat = VARIANTS[variant]
        self.variant = variant
        self.system_prompt = system
        self.num_shots = shots if num_shots is None else num_shots
        self.parser_name = parser
        self.max_new_tokens = tokens if max_new_tokens is None else max_new_tokens
        self.use_chat_template = use_chat
        self.date_string = date_string
        self.ctx_token_budget = ctx_token_budget
        self.pad_to_multiple_of = pad_to_multiple_of
        self.parse_stats = {}
        self.last_contexts = []
        self.last_retrieved = []

    @property
    def assembly_stats(self):
        """Overridden by AdaptiveAssemblyRAG; empty for the fixed top-k variants."""
        return {}

    def reset_stats(self):
        pass

    # -- prompt construction -------------------------------------------------

    def format_passages(self, passages):
        text = "\n".join(f"Title: {p['title']}\nPassage: {p['text']}" for p in passages)
        if self.ctx_token_budget is not None:
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) > self.ctx_token_budget:
                text = self.tokenizer.decode(ids[:self.ctx_token_budget],
                                             clean_up_tokenization_spaces=False)
        return text

    def build_messages(self, question, passages):
        messages = [{"role": "system", "content": self.system_prompt}]
        for user_text, gold, cot in EXEMPLARS[:self.num_shots]:
            messages.append({"role": "user", "content": user_text})
            if self.parser_name == "cot":
                messages.append({"role": "assistant", "content": cot})
            elif self.parser_name == "json":
                messages.append({"role": "assistant", "content": json.dumps({"answer": gold})})
            else:
                messages.append({"role": "assistant", "content": gold})
        messages.append({"role": "user",
                         "content": USER_TEMPLATE.format(passages=self.format_passages(passages),
                                                         question=question)})
        return messages

    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        list_passages, _ = self.search(queries, qids, k=k)
        self.last_retrieved = list_passages
        self.last_contexts = list_passages
        texts = []
        for query, passages in zip(queries, list_passages):
            if not self.use_chat_template:
                texts.append(f"{self.format_passages(passages)}\nQuestion: {query}\nAnswer:")
                continue
            texts.append(self.tokenizer.apply_chat_template(
                self.build_messages(query, passages),
                tokenize=False,
                add_generation_prompt=True,
                date_string=self.date_string,
            ))
        return texts

    # -- generation ----------------------------------------------------------

    def generation_kwargs(self, **kwargs):
        """Force greedy decoding and the variant's own budget.

        eval_for_rag hardcodes max_new_tokens=10 and passes the GPT-2
        eos_token_id as pad_token_id; both are overridden here.
        """
        kwargs = dict(kwargs)
        kwargs["max_new_tokens"] = self.max_new_tokens
        kwargs["do_sample"] = False
        kwargs["temperature"] = None
        kwargs["top_p"] = None
        kwargs["top_k"] = None
        kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        # Llama-3.2-Instruct stops on <|eot_id|> as well as <|end_of_text|>
        stop_ids = {self.tokenizer.eos_token_id}
        for token in ("<|eot_id|>", "<|eom_id|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id != self.tokenizer.unk_token_id:
                stop_ids.add(token_id)
        stop_ids.discard(None)
        kwargs["eos_token_id"] = sorted(stop_ids)
        return kwargs

    @torch.no_grad()
    def retrieval_augmented_generate(self, queries, qids, k=5, **kwargs):
        input_texts = self.make_augmented_inputs_for_generate(queries, qids, k=k)
        self.tokenizer.padding_side = "left"
        # apply_chat_template already emits <|begin_of_text|>: do not add a second BOS
        inputs = self.tokenizer(
            input_texts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=not self.use_chat_template,
            return_token_type_ids=False,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        outputs = self.model.generate(**inputs, **self.generation_kwargs(**kwargs))
        return outputs[:, inputs["input_ids"].size(1):]

    # -- post-processing -----------------------------------------------------

    def postprocess(self, predictions):
        parser = PARSERS[self.parser_name]
        self.parse_stats = {}
        parsed = [parser(pred, stats=self.parse_stats) for pred in predictions]
        self.parse_stats["n"] = len(predictions)
        self.parse_stats["mean_pred_words"] = (
            sum(len(p.split()) for p in parsed) / len(parsed) if parsed else 0.0
        )
        self.parse_stats["pct_empty"] = (
            sum(1 for p in parsed if not p.strip()) / len(parsed) if parsed else 0.0
        )
        return parsed
