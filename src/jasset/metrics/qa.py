"""QA content accuracy and format-aware accuracy from judge verdicts."""

import re

MTVQA_REVISION = "efc6a78a701516a76649430e1100c98e2904a7d8"


def mtvqa_score(prediction: str, annotations: str | list[str]) -> float:
    """Equivalent to pinned official scripts/acc.py, including substring semantics."""
    if isinstance(annotations, str):
        annotations = [annotations]
    if not annotations:
        raise ValueError("MTVQA requires at least one annotation")
    answer = prediction.strip().lower().replace(".", "")
    return float(any(a.strip().lower() in answer for a in annotations))


def extract_boxed(text: str) -> str | None:
    """Last complete balanced box; unlike reference code, no unboxed fallback."""
    result = None
    cursor = 0
    while (start := text.find(r"\boxed{", cursor)) >= 0:
        pos = start + len(r"\boxed{")
        begin, depth = pos, 1
        while pos < len(text) and depth:
            depth += (text[pos] == "{") - (text[pos] == "}")
            pos += 1
        if depth == 0:
            result = text[begin : pos - 1].strip()
        cursor = pos
    return result


def parse_verdict(text: str) -> float | None:
    match = re.fullmatch(r"\s*correct\s*:\s*(yes|no)\s*", text, re.IGNORECASE)
    return float(match[1].lower() == "yes") if match else None
