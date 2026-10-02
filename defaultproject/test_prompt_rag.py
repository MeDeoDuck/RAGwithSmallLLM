"""Unit tests for the RAG answer-parsing chain (no GPU, no model, no Lucene index).

    python test_prompt_rag.py
"""

import sys

from rag_parsing import parse_cot, parse_json, parse_plain

try:  # utils.metrics imports torch, which is not needed for the parsing tests
    from utils.metrics import normalize_answer
except ImportError:
    normalize_answer = None

CASES_PLAIN = [
    ("Marie Curie", "Marie Curie", "happy path"),
    ("\n\nMarie Curie.", "Marie Curie", "leading newlines and trailing period"),
    ("The answer is Marie Curie.", "Marie Curie", "restated label"),
    ("Answer: Marie Curie", "Marie Curie", "explicit label"),
    ("Marie Curie\nQuestion: who invented radium", "Marie Curie", "hallucinated next turn"),
    ("Sure! **Marie Curie**", "Marie Curie", "greeting and markdown"),
    ("- Marie Curie", "Marie Curie", "bullet"),
    ("Marie Curie, who was a Polish physicist", "Marie Curie", "trailing justification"),
    ("", "", "empty generation"),
    ("I don't know.", "I don't know", "abstention is passed through, not blanked"),
    ("OK Computer", "OK Computer", "greeting word that is really part of the answer"),
    ("Sure Thing", "Sure Thing", "same, second form"),
    ("Sure, Marie Curie", "Marie Curie", "real greeting, comma separated"),
    ("**Answer:** Titan", "Titan", "bold label"),
    ("**Answer**: Titan", "Titan", "bold label, colon outside"),
    ("Answer: **Titan**", "Titan", "bold answer"),
    ("\nQuestion: what is x", "Question: what is x",
     "generation starts with a turn marker: must not become empty"),
]

CASES_JSON = [
    ('{"answer": "Titan"}', "Titan", "happy path"),
    ('```json\n{"answer": "Titan"}\n```', "Titan", "code fence"),
    ('{"answer": "Titan"', "Titan", "missing closing brace"),
    ("{'answer': 'Titan'}", "Titan", "single quotes"),
    ('{"answer": "The answer is Titan"}', "Titan", "label inside the value"),
    ('{"Answer": ["Titan"]}', "Titan", "capitalised key, list value"),
    ("Titan", "Titan", "model abandoned JSON"),
    ("", "", "empty generation"),
]

CASES_COT = [
    ("Reasoning: The Titan passage says Titan is Saturn's largest moon.\nAnswer: Titan",
     "Titan", "happy path"),
    ("Let me think. The answer is Titan.", "Titan", "marker only"),
    ("Answer: Titan, which is also the second-largest moon.", "Titan", "trailing clause"),
    ("", "", "empty generation"),
]

CASES_COT_NONEMPTY = [
    "Reasoning: Saturn has many moons and the passage about Titan states that it",
    "Reasoning: first line\nsecond line without a marker",
    "\nQuestion: what is x",
    "<|eot_id|>Titan",
    "\nAnswer:",
    "\n\nTitle: Saturn\nPassage: ...",
    "   \n  \t ",
]

STATS_CASES = [
    (parse_plain, "Marie Curie"),
    (parse_plain, "\nQuestion: x"),
    (parse_json, '{"answer": "Titan"}'),
    (parse_json, '{"answer": "Titan"'),
    (parse_json, '{"answer": ""}'),
    (parse_json, "Titan"),
    (parse_cot, "Reasoning: a\nAnswer: Titan"),
    (parse_cot, "Reasoning: truncated mid sentence"),
]


def run():
    failures = []

    for parser, cases, label in [(parse_plain, CASES_PLAIN, "plain"),
                                 (parse_json, CASES_JSON, "json"),
                                 (parse_cot, CASES_COT, "cot")]:
        for raw, expected, description in cases:
            got = parser(raw)
            if got != expected:
                failures.append(f"[{label}] {description}: expected {expected!r}, got {got!r}")

    # C-5 invariant: a non-empty generation must never parse to an empty prediction,
    # because best_subspan_exact_match scores an empty prediction as wrong by definition
    all_raw = [raw for raw, _, _ in CASES_PLAIN + CASES_JSON + CASES_COT] + CASES_COT_NONEMPTY
    for parser in (parse_plain, parse_json, parse_cot):
        for raw in all_raw:
            if raw.strip() and not parser(raw).strip():
                failures.append(f"[{parser.__name__}] empty output for non-empty input {raw!r}")

    # every prediction must be counted exactly once, or the reported diagnostics
    # will not add up to n
    for parser, raw in STATS_CASES:
        stats = {}
        parser(raw, stats=stats)
        if sum(stats.values()) != 1:
            failures.append(f"[{parser.__name__}] {raw!r} counted {sum(stats.values())} times: {stats}")

    # the normalizer used by the metric must not turn a parsed answer into nothing
    if normalize_answer is not None:
        for raw, expected, _ in CASES_PLAIN + CASES_JSON + CASES_COT:
            if expected and not normalize_answer(parse_plain(raw) or expected):
                failures.append(f"normalize_answer blanked {expected!r}")
    else:
        print("note: torch is missing, skipping the normalize_answer check")

    total = len(CASES_PLAIN) + len(CASES_JSON) + len(CASES_COT)
    if failures:
        print(f"FAILED {len(failures)} check(s) out of {total} cases:")
        for failure in failures:
            print("  -", failure)
        return 1
    print(f"OK: {total} parsing cases and the non-empty invariant passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
