"""Text normalisation for each metric. Each metric uses exactly one of these."""

import re
import unicodedata

CCOCR_REVISION = "fb6187ebe5374d5682ac0bc05f31e3639ab6fb15"

__all__ = [
    "CCOCR_REVISION",
    "ccocr_tokens",
    "line_text",
    "normalize_cer",
    "normalize",
    "is_cjk",
]


def normalize(text: str) -> str:
    """Strict string normalization used by archived scoring and rewards."""
    return "".join(unicodedata.normalize("NFKC", text).split())


def line_text(text):
    """NED and region OCR: NFKC, then remove all whitespace."""
    return normalize(str(text))


def ccocr_tokens(text):
    """Character P/R/F1: CC-OCR ocr_evaluator.py tokens (no NFKC)."""
    text = str(text).strip().replace("\t", " ").replace("\n", " ")
    text = text.replace("###", "").replace("***", "")
    text = re.sub(r"\s+", " ", text).replace(" ", "")
    return [char.lower() for char in text]


_CJK_RANGES = (
    (0x2E80, 0x2FDF),  # CJK radicals, Kangxi radicals
    (0x3000, 0x303F),  # CJK symbols and punctuation
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x3190, 0x31FF),  # Kanbun, Bopomofo ext, Katakana phonetic ext
    (0x3200, 0x33FF),  # enclosed CJK, CJK compatibility
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK unified
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFE30, 0xFE4F),  # CJK compatibility forms
    (0xFF00, 0xFFEF),  # halfwidth and fullwidth forms
    (0x20000, 0x3134F),  # CJK ext B-G
)


def is_cjk(char):
    code = ord(char)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def normalize_cer(text):
    """nu_cer: NFKC, newlines to spaces, collapse runs, keep a space only between non-CJK."""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    out = []
    for i, ch in enumerate(text):
        if ch == " ":
            if i == 0 or i == len(text) - 1:
                continue
            if is_cjk(text[i - 1]) or is_cjk(text[i + 1]):
                continue
        out.append(ch)
    return "".join(out)
