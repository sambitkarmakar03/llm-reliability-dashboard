"""Heuristic detection of non-answers: refusals, knowledge-cutoff
disclaimers, and similar text that makes no verifiable claim and
therefore shouldn't be scored as if it were a real answer.
"""

REFUSAL_PATTERNS = [
    "i don't have information",
    "i do not have information",
    "i don't have access to",
    "i do not have access to",
    "my knowledge cutoff",
    "my training data",
    "i'm not aware of",
    "i am not aware of",
    "i don't know",
    "i do not know",
    "as an ai",
    "i cannot provide",
    "i can't provide",
    "no information available",
    "i don't have real-time",
    "i do not have real-time",
    "i'm unable to",
    "i am unable to",
    "beyond my knowledge",
    "i have no information",
]


def is_refusal(text: str, max_chars: int = 300) -> bool:
    """
    True if this looks like a refusal/disclaimer rather than an actual
    answer. Checked against a prefix of the text since these disclaimers
    are almost always stated up front, not buried mid-response.
    """
    if not text:
        return False
    snippet = text[:max_chars].lower()
    return any(pattern in snippet for pattern in REFUSAL_PATTERNS)