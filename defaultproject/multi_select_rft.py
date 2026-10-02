"""Multi-passage selection: pick any subset of the five, including none.

    {"sources": [2, 4]}        two passages
    {"sources": []}            no passage carries the evidence

Reward is the Dice overlap between the selected set S and the label set T:

    format error      -1
    n = 0, k = 0       1
    n = 0, k > 0       0
    n > 0            2*TP / (n + k)

What this reward asks for, stated plainly: **select every passage whose text
contains a gold alias**. It is not a reward for finding minimal sufficient
evidence. With T = {2, 4} the full set [2, 4] scores 1.0 while [2] scores 0.67,
even in the common case where passage 2 alone is enough to answer. A policy that
learns to add a second redundant copy of the same fact is doing exactly what it
was paid to do.

The label itself is a string test -- a normalised gold alias occurs in the
passage text. It does not verify that the passage supports the answer, and a
passage can support the answer without containing the alias.
"""
import json
import re

from utils.metrics import normalize_answer

SLOTS = 5
REWARD_MALFORMED = -1.0
REWARD_ABSTAIN_CORRECT = 1.0
REWARD_ZERO = 0.0

SYS_MULTI = """You are given a question and five numbered Wikipedia passages.

Your task is to select every passage that contains evidence needed to answer the question.

Rules:
- Reply with JSON only, in exactly this form: {"sources": [1, 3]}
- List the numbers of the passages you select, in increasing order, with no repeats.
- Each number must be between 1 and 5.
- Use {"sources": []} if none of the five passages contains the evidence.
- Do not add any other key. Do not add any text before or after the JSON. Do not answer the question."""

EXEMPLAR_MULTI_USER = """[1] Title: Saturn
Passage: Saturn is the sixth planet from the Sun and the second-largest in the Solar System, after Jupiter.
[2] Title: Titan (moon)
Passage: Titan is the largest moon of Saturn and the second-largest natural satellite in the Solar System.
[3] Title: Jupiter
Passage: Jupiter is the fifth planet from the Sun and the largest in the Solar System.
[4] Title: Moons of Saturn
Passage: Saturn has 146 confirmed moons; the largest of them is Titan, which is bigger than the planet Mercury.
[5] Title: Solar System
Passage: The Solar System is the gravitationally bound system of the Sun and the objects that orbit it.

Question: what is the largest moon of saturn"""
EXEMPLAR_MULTI_GOLD = {"sources": [2, 4]}

EXEMPLAR_EMPTY_USER = """[1] Title: Association football
Passage: Association football, more commonly known as football or soccer, is a team sport played between two teams of 11 players.
[2] Title: FIFA World Cup
Passage: The FIFA World Cup is an international association football competition contested by the senior men's national teams.
[3] Title: Offside (association football)
Passage: Offside is one of the laws of association football, codified in Law 11 of the Laws of the Game.
[4] Title: Penalty kick
Passage: A penalty kick is a method of restarting play in association football.
[5] Title: Football pitch
Passage: A football pitch is the playing surface for the game of association football.

Question: who won the 1994 fifa world cup final"""
EXEMPLAR_EMPTY_GOLD = {"sources": []}

# The abstention demonstration comes first and the selection second. With the
# order reversed, the untrained policy answered {"sources": []} on 71.7% of the
# dev questions that did have an answer-bearing passage, and 78.8% of GRPO
# groups had every sample identical -- a few-shot model copies the last
# demonstration hardest, and the last one was the empty list.
MULTI_EXEMPLARS = [(EXEMPLAR_EMPTY_USER, EXEMPLAR_EMPTY_GOLD),
                   (EXEMPLAR_MULTI_USER, EXEMPLAR_MULTI_GOLD)]

DATE_STRING = "26 Jul 2024"


# ---------------------------------------------------------------------- labels

def passage_flags(passages, answers):
    """g_i -- a normalised gold alias occurs in passage i. A string test."""
    norm = [a for a in (normalize_answer(x) for x in answers) if a]
    flags = []
    for passage in passages:
        blob = normalize_answer("%s %s" % (passage.get("title", ""), passage.get("text", "")))
        flags.append(int(any(a in blob for a in norm)))
    return flags


def label_set(flags):
    """T -- the 1-based indices of the answer-bearing passages."""
    return {i + 1 for i, g in enumerate(flags) if g}


# ---------------------------------------------------------------------- prompt

def format_numbered_passages(passages):
    return "\n".join("[%d] Title: %s\nPassage: %s" % (i, p.get("title", ""), p.get("text", ""))
                     for i, p in enumerate(passages, 1))


def build_messages(question, passages, num_shots=2):
    messages = [{"role": "system", "content": SYS_MULTI}]
    for user_text, gold in MULTI_EXEMPLARS[:num_shots]:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": json.dumps(gold)})
    messages.append({"role": "user",
                     "content": "%s\n\nQuestion: %s"
                                % (format_numbered_passages(passages), question)})
    return messages


def build_prompt(tokenizer, question, passages, num_shots=2):
    return tokenizer.apply_chat_template(
        build_messages(question, passages, num_shots),
        tokenize=False, add_generation_prompt=True, date_string=DATE_STRING)


# --------------------------------------------------------------------- parsing

def parse_sources(text, slots=SLOTS):
    """-> (sources, valid). `sources` is a list; [] is a legal selection.

    Strict on purpose. Duplicates and out-of-order lists are format errors, not
    something to clean up: silently sorting the model's output would mean the
    reward is computed on tokens the model never produced, and the policy
    gradient would then be taken against a rewritten sequence.
    """
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        return None, False
    try:
        obj = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None, False
    if not isinstance(obj, dict) or "sources" not in obj:
        return None, False
    raw = obj["sources"]
    if not isinstance(raw, list):
        return None, False
    if len(raw) > slots:
        return None, False
    picked = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            return None, False
        if not 1 <= value <= slots:
            return None, False
        picked.append(value)
    if len(set(picked)) != len(picked):
        return None, False                      # duplicates
    if picked != sorted(picked):
        return None, False                      # ascending order required
    return picked, True


# ---------------------------------------------------------------------- reward

def reward(sources, flags):
    """Dice overlap of the selection with the label set. No answer involved."""
    target = label_set(flags)
    chosen = set(sources)
    n, k = len(target), len(chosen)
    if n == 0:
        return REWARD_ABSTAIN_CORRECT if k == 0 else REWARD_ZERO
    hit = len(target & chosen)
    return 2.0 * hit / (n + k)


def score_candidate(text, flags):
    sources, valid = parse_sources(text)
    if not valid:
        return {"text": text, "valid": False, "sources": None,
                "reward": REWARD_MALFORMED, "k": None, "tp": None}
    target = label_set(flags)
    return {"text": text, "valid": True, "sources": sources,
            "reward": reward(sources, flags),
            "k": len(sources), "tp": len(target & set(sources))}


# -------------------------------------------------------------------- metrics

def selection_scores(sources, flags):
    """Per-question precision, recall and F1, plus the n = 0 abstain outcome.

    On the empty set the ratios have no denominator, so they are reported as
    None rather than as 0 or 1: folding an abstention into a mean F1 would make
    the headline move for a reason that has nothing to do with ranking quality.
    `n = 0` questions are scored separately by whether the model abstained.
    """
    target = label_set(flags)
    chosen = set(sources)
    n, k = len(target), len(chosen)
    hit = len(target & chosen)
    out = {"n": n, "k": k, "tp": hit, "exact": target == chosen,
           "precision": None, "recall": None, "f1": None, "abstain_correct": None}
    if n == 0:
        out["abstain_correct"] = (k == 0)
        return out
    out["recall"] = hit / n
    out["precision"] = (hit / k) if k else None
    out["f1"] = 2.0 * hit / (n + k)             # equals Dice; 0 when k = 0
    return out
