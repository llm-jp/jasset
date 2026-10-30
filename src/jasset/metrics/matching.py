"""Minimum-edit line assignment; inputs are already normalized strings.

Reference order and the dummy layout are part of the frozen scoring protocol.
This minimizes edits, without a secondary exact-match objective.
"""


def edit_distance(a: str, b: str) -> int:
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(a, b)


def minimum_edit_assignment(refs, preds):
    from rapidfuzz.distance import Levenshtein
    from scipy.optimize import linear_sum_assignment

    n, m = len(refs), len(preds)
    # Every real row can use any deletion dummy; every real column any insertion
    # dummy. Dummy-to-dummy zero entries permit all partial matching patterns.
    costs = [[0] * (n + m) for _ in range(n + m)]
    for i, ref in enumerate(refs):
        for j, pred in enumerate(preds):
            costs[i][j] = Levenshtein.distance(ref, pred)
        for j in range(m, m + n):
            costs[i][j] = len(ref)
    for i in range(n, n + m):
        for j, pred in enumerate(preds):
            costs[i][j] = len(pred)
    rows, cols = linear_sum_assignment(costs) if n + m else ([], [])
    return costs, rows, cols
