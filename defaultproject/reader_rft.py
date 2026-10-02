"""Output contract, parser and reward for the RFT reader.

Sampling, training and evaluation all import from here so the three cannot drift
apart: a reward computed during rollout collection must be the same number the
evaluation would produce for that output.

The reader is asked to name the passage it used:

    {"answer": "Cyndi Lauper", "source": 3}

`source` makes the two behavioural reward terms measurable. Without it the only
thing an output reveals is the answer string, and a term the policy cannot move
contributes no gradient.

Measurement caveat, repeated here because it is easy to forget at the call site:
`g_i` is a *string* test -- does a normalised gold alias occur in the passage.
It is not evidence that the passage supports the answer (a common alias such as
"1975" matches by accident), nor does `g_i = 0` mean the passage is unsupportive
(a passage can establish the answer without containing the alias).
"""
import json
import re

from utils.metrics import normalize_answer

SLOTS = 5
ALPHA = 0.3                     # correct *and* cited an answer-bearing passage
BETA = 0.5                      # cited a passage with no gold alias in it


# ---------------------------------------------------------------------- prompt
# v3's contract plus one field. The passages have to carry visible numbers for
# `source` to refer to anything, so this variant changes two things at once
# relative to v3: the numbering and the extra key. G1 measures their combined
# effect; the two are not separated.

SYS_SOURCE = """You are a question answering system. You are given numbered Wikipedia passages and one question.

Rules:
- Use the passages to answer. The answer is a span of text, usually a name, a date, a number, or a short title.
- The answer is at most 5 words.
- Do not write a sentence. Do not repeat the question. Do not explain your choice.
- If the passages do not contain the answer, give your single best guess in the same short form. Never reply "unknown", "not stated", "not mentioned", or "I don't know".
- Output format: one line of JSON and nothing else, in exactly this form:
{"answer": "<the answer>", "source": <the number of the passage you used>}
"source" must be one of the passage numbers shown. Do not wrap the JSON in a code block. Do not add any other key. Do not add any text before or after the JSON."""

# the answer sits in passage 2 in the first exemplar and passage 1 in the
# second, so the shots do not demonstrate a constant source
EXEMPLAR_A_USER = """[1] Title: Saturn
Passage: Saturn is the sixth planet from the Sun and the second-largest in the Solar System, after Jupiter. It is a gas giant with an average radius about nine and a half times that of Earth.
[2] Title: Titan (moon)
Passage: Titan is the largest moon of Saturn and the second-largest natural satellite in the Solar System. It is the only moon known to have a dense atmosphere.

Question: what is the largest moon of saturn"""
EXEMPLAR_A_GOLD = {"answer": "Titan", "source": 2}

EXEMPLAR_B_USER = """[1] Title: Marie Curie
Passage: Marie Curie was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity. In 1903 she became the first woman to win a Nobel Prize, sharing the Nobel Prize in Physics with her husband Pierre Curie and Henri Becquerel.
[2] Title: Pierre Curie
Passage: Pierre Curie was a French physicist and a pioneer in crystallography, magnetism, piezoelectricity and radioactivity.

Question: who was the first woman to win a nobel prize"""
EXEMPLAR_B_GOLD = {"answer": "Marie Curie", "source": 1}

SOURCE_EXEMPLARS = [(EXEMPLAR_A_USER, EXEMPLAR_A_GOLD),
                    (EXEMPLAR_B_USER, EXEMPLAR_B_GOLD)]


DATE_STRING = "26 Jul 2024"     # matches PromptedRAG's default


def format_numbered_passages(passages):
    """`source` can only refer to a passage the prompt actually numbers."""
    return "\n".join("[%d] Title: %s\nPassage: %s" % (i, p.get("title", ""), p.get("text", ""))
                     for i, p in enumerate(passages, 1))


def build_messages(question, passages, num_shots=2):
    """The chat messages for one question. Single source for sampler and reader.

    Rollout collection and evaluation must agree token for token: a reward
    attached to a sample is only meaningful if the evaluated prompt is the same
    prompt the sample was drawn from.
    """
    messages = [{"role": "system", "content": SYS_SOURCE}]
    for user_text, gold in SOURCE_EXEMPLARS[:num_shots]:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": json.dumps(gold)})
    messages.append({"role": "user",
                     "content": "%s\n\nQuestion: %s"
                                % (format_numbered_passages(passages), question)})
    return messages


def build_prompt(tokenizer, question, passages, num_shots=2):
    """Rendered prompt string, generation turn open."""
    return tokenizer.apply_chat_template(
        build_messages(question, passages, num_shots),
        tokenize=False, add_generation_prompt=True, date_string=DATE_STRING)


# --------------------------------------------------------------------- labels

def passage_flags(passages, answers):
    """g_i for each passage: does a normalised gold alias occur in it?"""
    norm = [normalize_answer(a) for a in answers]
    norm = [a for a in norm if a]
    flags = []
    for passage in passages:
        blob = normalize_answer("%s %s" % (passage.get("title", ""), passage.get("text", "")))
        flags.append(int(any(a in blob for a in norm)))
    return flags


def answer_count(passages, answers):
    """m -- computed from the passages actually handed to the model, never a pool."""
    return sum(passage_flags(passages, answers))


# --------------------------------------------------------------------- parsing

def parse_output(text, slots=SLOTS):
    """-> (answer, source_index_1based, valid).

    `valid` gates everything downstream: an invalid candidate gets no reward and
    is excluded from the group statistics, rather than being scored as very bad.
    Scoring it would let the format-error rate move the group mean.
    """
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        return "", None, False
    try:
        obj = json.loads(match.group(0))
    except (ValueError, TypeError):
        return "", None, False
    if not isinstance(obj, dict):
        return "", None, False
    answer = obj.get("answer")
    source = obj.get("source")
    if not isinstance(answer, str):
        return "", None, False
    if isinstance(source, str) and source.strip().lstrip("[").rstrip("]").isdigit():
        source = int(source.strip().lstrip("[").rstrip("]"))
    if not isinstance(source, int) or isinstance(source, bool):
        return answer, None, False
    if not 1 <= source <= slots:
        return answer, None, False
    return answer, source, True


def is_correct(answer, answers):
    """best_subspan_exact_match for a single prediction; empty is always wrong."""
    pred = normalize_answer(answer)
    if not pred:
        return 0
    for gold in answers:
        gold = normalize_answer(gold)
        if gold and gold in pred:
            return 1
    return 0


# ---------------------------------------------------------------------- reward

def reward(answer, source, answers, flags):
    """R = e + ALPHA * e * g_s - BETA * (1 - g_s), for a valid output.

    Ordering this produces, on a question with m > 0:

        correct + cited g=1   1.3
        correct + cited g=0   0.5
        wrong   + cited g=1   0.0
        wrong   + cited g=0  -0.5

    The second must outrank the third: the metric we report counts the second as
    right and the third as wrong, so any ordering that inverts them optimises
    against our own measure. That is what fixes BETA < 1.
    """
    e = is_correct(answer, answers)
    g = flags[source - 1]
    return e + ALPHA * e * g - BETA * (1 - g)


def score_candidate(text, answers, flags):
    """One rollout -> the record the sampler writes out."""
    answer, source, valid = parse_output(text)
    if not valid:
        return {"text": text, "valid": False, "answer": answer,
                "source": None, "e": None, "g": None, "reward": None}
    e = is_correct(answer, answers)
    g = flags[source - 1]
    return {"text": text, "valid": True, "answer": answer, "source": source,
            "e": e, "g": g, "reward": e + ALPHA * e * g - BETA * (1 - g)}


# ------------------------------------------------------------------- advantage

def group_advantage(records, eps=1e-6):
    """Group-relative advantage over the *valid* candidates only.

    Returns (advantages_by_index, reason) where reason is None when the group is
    usable and otherwise names why it was skipped. The caller logs those reasons:
    with one- and two-word answers a group can easily sample the same output
    eight times, and a run where most groups are degenerate has nothing to learn
    from no matter how long it is left going.
    """
    valid = [i for i, r in enumerate(records) if r["valid"]]
    if not valid:
        return {}, "no_valid"
    rewards = [records[i]["reward"] for i in valid]
    mean = sum(rewards) / len(rewards)
    var = sum((x - mean) ** 2 for x in rewards) / len(rewards)
    std = var ** 0.5
    if std == 0.0:
        return {}, "zero_variance"
    adv = {i: (records[i]["reward"] - mean) / (std + eps) for i in valid}
    keep = {i: a for i, a in adv.items() if a > 0 and records[i]["e"] == 1}
    if not keep:
        return {}, "no_correct_candidate"
    total = sum(keep.values())
    return {i: a / total for i, a in keep.items()}, None
