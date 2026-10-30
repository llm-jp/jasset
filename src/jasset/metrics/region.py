"""Region OCR: normalised edit distance and exact match of one cropped region."""

from rapidfuzz.distance import Levenshtein

from jasset.metrics.normalize import line_text


def region_scores(reference, prediction):
    ref, out = line_text(reference), line_text(prediction)
    if not ref:
        raise ValueError("Nonempty normalized reference required")
    return dict(
        ned=Levenshtein.distance(ref, out) / max(len(ref), len(out)), em=ref == out
    )
