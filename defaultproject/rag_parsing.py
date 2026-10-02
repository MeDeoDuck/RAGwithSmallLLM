"""Answer parsing for the prompted RAG variants (no torch / pyserini import).

Kept in its own module so the parsing chain can be unit-tested without a GPU,
an LLM or the Lucene index.
"""

import json
import re

# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_TURN_MARKERS = ["\nquestion:", "\nq:", "\ntitle:", "\npassage:", "\nuser", "\nassistant",
                 "\nnote:", "\nexplanation:", "\nsource:"]

# the trailing punctuation is required: "OK Computer" and "Sure Thing" are valid NQ answers
_GREETING_RE = re.compile(r"^\s*(sure|okay|ok|of course|certainly|here(?:'s| is)(?: the)?(?: answer)?)"
                          r"\s*[,:.!\-\n]+\s*", re.IGNORECASE)
_LABEL_RE = re.compile(r"^\s*(?:the\s+)?(?:final\s+|short\s+|correct\s+)?answer(?:\s+is)?\s*[:\-–—]?\s*",
                       re.IGNORECASE)
_SHORT_LABEL_RE = re.compile(r"^\s*A\s*[:.]\s*")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_TRAILING_CLAUSE_RE = re.compile(r"\s*[,(]\s*(which|who|as |according to|based on|per )\b.*$",
                                 re.IGNORECASE | re.DOTALL)
_QUOTES = "\"'`‘’“”"

MAX_ANSWER_WORDS = 12
MAX_FALLBACK_WORDS = 20


def clean_raw(raw):
    """C-0: strip special-token residue and any hallucinated next turn."""
    if raw is None:
        return ""
    text = _SPECIAL_TOKEN_RE.sub(" ", raw).replace("�", "")
    lowered = text.lower()
    cut = len(text)
    for marker in _TURN_MARKERS:
        pos = lowered.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    return text[:cut].strip()


def _residual(raw):
    """What survives when the turn-marker cut of clean_raw removed everything."""
    if raw is None:
        return ""
    return _SPECIAL_TOKEN_RE.sub(" ", raw).replace("�", "").strip()


def _first_nonempty_line(text):
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


def _last_nonempty_line(text):
    for line in reversed(text.splitlines()):
        if line.strip():
            return line
    return ""


def unlabel(text):
    """C-1: strip greetings, answer labels, markdown and trailing justifications."""
    if not text:
        return ""
    s = _GREETING_RE.sub("", text, count=1)
    # markdown first: "**Answer:** Titan" must lose its asterisks before the label is matched
    s = s.replace("**", "")
    s = _BULLET_RE.sub("", s, count=1)
    for _ in range(2):
        new = _LABEL_RE.sub("", s, count=1)
        new = _SHORT_LABEL_RE.sub("", new, count=1)
        if new == s:
            break
        s = new
    s = s.strip().strip(_QUOTES).strip()
    s = _TRAILING_CLAUSE_RE.sub("", s)
    s = " ".join(s.split()[:MAX_ANSWER_WORDS])
    s = s.rstrip(" .,;:!" + _QUOTES)
    return s.strip()


def _final_fallback(cleaned):
    """C-5: never return an empty string when the model produced something."""
    line = _first_nonempty_line(cleaned).strip()
    if not line:
        return ""
    return " ".join(line.split()[:MAX_FALLBACK_WORDS])


def parse_plain(raw, stats=None):
    """C-2: parsing chain for the terse variants (V0/V1/V2)."""
    cleaned = clean_raw(raw)
    if not cleaned:
        residual = _residual(raw)
        if not residual:
            _bump(stats, "empty")
            return ""
        # the whole generation was a hallucinated next turn: keep something non-empty
        _bump(stats, "clean_empty")
        return _final_fallback(residual)
    answer = unlabel(_first_nonempty_line(cleaned))
    if answer:
        _bump(stats, "format_ok")
        return answer
    _bump(stats, "fallback")
    return _final_fallback(cleaned)


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_JSON_QUOTED_RE = re.compile(r"[\"']answer[\"']\s*:\s*[\"'](.*?)[\"']", re.IGNORECASE | re.DOTALL)
_JSON_BARE_RE = re.compile(r"[\"']?answer[\"']?\s*:\s*([^,}\n]+)", re.IGNORECASE)


def parse_json(raw, stats=None):
    """C-3: parsing chain for the JSON variant (V3)."""
    cleaned = clean_raw(raw)
    if not cleaned:
        residual = _residual(raw)
        if not residual:
            _bump(stats, "empty")
            return ""
        # the whole generation was a hallucinated next turn: keep something non-empty
        _bump(stats, "clean_empty")
        return _final_fallback(residual)

    s = re.sub(r"^\s*```[a-zA-Z]*\s*", "", cleaned)
    s = re.sub(r"\s*```\s*$", "", s).strip()

    value = None
    strict = False
    match = _JSON_OBJ_RE.search(s)
    if match:
        try:
            obj = json.loads(match.group(0))
        except (ValueError, TypeError):
            obj = None
        if isinstance(obj, dict):
            lookup = {str(key).casefold(): val for key, val in obj.items()}
            for key in ("answer", "ans", "a"):
                if key in lookup:
                    value = lookup[key]
                    break
            if isinstance(value, list):
                value = value[0] if value else None
            if value is not None and not isinstance(value, str):
                value = str(value)
            strict = value is not None

    if value is None:
        quoted = _JSON_QUOTED_RE.search(s)
        if quoted:
            value = quoted.group(1)
    if value is None:
        bare = _JSON_BARE_RE.search(s)
        if bare:
            value = bare.group(1)
    if value is None:
        _bump(stats, "json_abandoned")
        return parse_plain(raw, stats=None)

    value = value.replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\")
    answer = unlabel(value)
    if answer:
        _bump(stats, "format_ok" if strict else "json_repaired")
        return answer
    _bump(stats, "fallback")
    return _final_fallback(cleaned)


_COT_MARKERS = [
    re.compile(r"(?im)^\s*answer\s*[:\-]\s*"),
    re.compile(r"(?i)\bfinal answer\s*(?:is)?\s*[:\-]?\s*"),
    re.compile(r"(?i)\bso[, ]+the answer is\b\s*"),
    re.compile(r"(?i)\bthe answer is\b\s*"),
]
_THINK_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)


def parse_cot(raw, stats=None):
    """C-4: parsing chain for the CoT variant (V4)."""
    cleaned = clean_raw(raw)
    if not cleaned:
        residual = _residual(raw)
        if not residual:
            _bump(stats, "empty")
            return ""
        # the whole generation was a hallucinated next turn: keep something non-empty
        _bump(stats, "clean_empty")
        return _final_fallback(residual)

    think = list(_THINK_RE.finditer(cleaned))
    if think:
        cleaned = cleaned[think[-1].end():].strip()

    for marker in _COT_MARKERS:
        matches = list(marker.finditer(cleaned))
        if matches:
            tail = cleaned[matches[-1].end():]
            answer = unlabel(_first_nonempty_line(tail))
            if answer:
                _bump(stats, "format_ok")
                return answer

    # no "Answer:" marker: the generation was truncated mid-reasoning
    body = "\n".join(line for line in cleaned.splitlines()
                     if not re.match(r"^\s*reasoning\s*[:\-]", line, re.IGNORECASE))
    answer = unlabel(_last_nonempty_line(body) or _last_nonempty_line(cleaned))
    if answer:
        _bump(stats, "cot_truncated")
        return answer
    _bump(stats, "fallback")
    return _final_fallback(cleaned)


PARSERS = {
    "plain": parse_plain,
    "json": parse_json,
    "cot": parse_cot,
}


def _bump(stats, key):
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1
