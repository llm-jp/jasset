"""Exact multiset F1 and maximum-similarity one-to-one soft F1.

This assignment maximizes normalized similarity, unlike NED's minimum edit cost.
Caps and repetition penalties belong to the reward policy, not these metrics.
"""

from collections import Counter

from jasset.metrics.matching import edit_distance
from jasset.metrics.normalize import normalize


def set_scores(prediction, references):
    refs = [normalize(x) for x in references]
    if not refs or not all(refs):
        raise ValueError("Nonempty reference items required")
    preds = [normalize(x) for x in prediction.splitlines() if normalize(x)]
    return normalized_set_scores(refs, preds)


def normalized_set_scores(refs, preds):
    """Score already normalized nonempty items without normalizing them again."""
    if not refs or not all(refs) or not all(preds):
        raise ValueError("Nonempty normalized items required")
    from scipy.optimize import linear_sum_assignment

    denominator = len(refs) + len(preds)
    exact = 2 * sum((Counter(refs) & Counter(preds)).values()) / denominator
    similarity = [
        [1 - edit_distance(a, b) / max(len(a), len(b)) for b in preds] for a in refs
    ]
    total = 0.0
    if preds:
        rows, cols = linear_sum_assignment(similarity, maximize=True)
        total = sum(similarity[i][j] for i, j in zip(rows, cols))
    soft = 2 * total / denominator
    return {
        "exact_f1": exact,
        "soft_f1": soft,
        "prediction_items": len(preds),
        "reference_items": len(refs),
    }
