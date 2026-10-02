"""Evidence selection as the trained task, with answer correctness kept out of it.

The policy sees a question and the five passages the existing pipeline already
chose, and emits one number:

    {"source": k}        k = 1..5 picks that passage, k = 0 picks none

The reward scores that choice against whether any of the five carries a gold
alias, and nothing else. Whether the reader then produces the right answer is an
evaluation outcome, not a training signal -- which is the only way "selection
improved" and "answers improved" can be told apart afterwards.

Label caveat: `n(q, D)` counts passages whose text contains a normalised gold
alias. It is a string test. A passage can match by accident (a bare year, a
common name) and a passage can support the answer without containing the alias.
Nothing here verifies semantic support, and the reward inherits that limit.
"""
import json
import re

from utils.metrics import normalize_answer

SLOTS = 5
ABSTAIN = 0

REWARD_GOOD = 1.0
REWARD_BAD = 0.0
REWARD_MALFORMED = -1.0

SYS_SELECT = """You are given a question and five numbered Wikipedia passages.

Your task is to pick the one passage that contains the evidence needed to answer the question.

Rules:
- Reply with JSON only, in exactly this form: {"source": <number>}
- Use 1, 2, 3, 4 or 5 to pick that passage.
- Use 0 if none of the five passages contains the evidence.
- Do not add any other key. Do not add any text before or after the JSON. Do not answer the question."""

# the exemplars cover one pick and one abstention so neither is the obvious reply
EXEMPLAR_PICK_USER = """[1] Title: Saturn
Passage: Saturn is the sixth planet from the Sun and the second-largest in the Solar System, after Jupiter.
[2] Title: Titan (moon)
Passage: Titan is the largest moon of Saturn and the second-largest natural satellite in the Solar System.
[3] Title: Jupiter
Passage: Jupiter is the fifth planet from the Sun and the largest in the Solar System.
[4] Title: Enceladus
Passage: Enceladus is the sixth-largest moon of Saturn, about 500 kilometres in diameter.
[5] Title: Solar System
Passage: The Solar System is the gravitationally bound system of the Sun and the objects that orbit it.

Question: what is the largest moon of saturn"""
EXEMPLAR_PICK_GOLD = {"source": 2}

EXEMPLAR_ABSTAIN_USER = """[1] Title: Association football
Passage: Association football, more commonly known as football or soccer, is a team sport played between two teams of 11 players.
[2] Title: FIFA World Cup
Passage: The FIFA World Cup is an international association football competition contested by the senior men's national teams.
[3] Title: Offside (association football)
Passage: Offside is one of the laws of association football, codified in Law 11 of the Laws of the Game.
[4] Title: Penalty kick
Passage: A penalty kick is a method of restarting play in association football, in which a player is allowed to take a shot on goal.
[5] Title: Football pitch
Passage: A football pitch is the playing surface for the game of association football.

Question: who won the 1994 fifa world cup final"""
EXEMPLAR_ABSTAIN_GOLD = {"source": 0}

SELECT_EXEMPLARS = [(EXEMPLAR_PICK_USER, EXEMPLAR_PICK_GOLD),
                    (EXEMPLAR_ABSTAIN_USER, EXEMPLAR_ABSTAIN_GOLD)]

DATE_STRING = "26 Jul 2024"


# ---------------------------------------------------------------------- labels

def passage_flags(passages, answers):
    """g_i -- does a normalised gold alias occur in passage i? A string test."""
    norm = [a for a in (normalize_answer(x) for x in answers) if a]
    flags = []
    for passage in passages:
        blob = normalize_answer("%s %s" % (passage.get("title", ""), passage.get("text", "")))
        flags.append(int(any(a in blob for a in norm)))
    return flags


def evidence_count(flags):
    """n(q, D) -- how many of the passages actually given to the model match."""
    return sum(flags)


# ---------------------------------------------------------------------- prompt

def format_numbered_passages(passages):
    return "\n".join("[%d] Title: %s\nPassage: %s" % (i, p.get("title", ""), p.get("text", ""))
                     for i, p in enumerate(passages, 1))


def build_messages(question, passages, num_shots=2):
    messages = [{"role": "system", "content": SYS_SELECT}]
    for user_text, gold in SELECT_EXEMPLARS[:num_shots]:
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

def parse_selection(text, slots=SLOTS):
    """-> (source, valid). `source` is 0..slots; None when the reply is unusable.

    Abstention is a real action here, so 0 is in range and must not be confused
    with a parse failure -- the two get different rewards.
    """
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        return None, False
    try:
        obj = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None, False
    if not isinstance(obj, dict):
        return None, False
    source = obj.get("source")
    if isinstance(source, str) and source.strip().lstrip("[").rstrip("]").isdigit():
        source = int(source.strip().lstrip("[").rstrip("]"))
    if not isinstance(source, int) or isinstance(source, bool):
        return None, False
    if not ABSTAIN <= source <= slots:
        return None, False
    return source, True


# ---------------------------------------------------------------------- reward

def reward(source, flags):
    """The four rules. Answer correctness is deliberately absent.

        n > 0, picked an answer-bearing passage      Good
        n > 0, picked another passage or abstained   Bad
        n = 0, abstained                             Good
        n = 0, picked a passage                      Bad
    """
    n = evidence_count(flags)
    if n > 0:
        good = source != ABSTAIN and flags[source - 1] == 1
    else:
        good = source == ABSTAIN
    return REWARD_GOOD if good else REWARD_BAD


def score_candidate(text, flags):
    source, valid = parse_selection(text)
    if not valid:
        return {"text": text, "valid": False, "source": None,
                "reward": REWARD_MALFORMED}
    return {"text": text, "valid": True, "source": source,
            "reward": reward(source, flags)}


def outcome(source, flags):
    """Reporting bucket for one selection, used by the evaluation split."""
    n = evidence_count(flags)
    if n > 0:
        if source == ABSTAIN:
            return "n>0_abstained"
        return "n>0_hit" if flags[source - 1] == 1 else "n>0_miss"
    return "n=0_abstained" if source == ABSTAIN else "n=0_picked"
