"""Full-image normalised edit distance with a minimum-cost one-to-one line assignment."""

from jasset.metrics.matching import minimum_edit_assignment
from jasset.metrics.normalize import line_text


def ned_scores(references, prediction):
    """Match references and output lines, then divide errors by item-wise upper bounds.

    A matched pair costs its edit distance (bound max(|t|, |l|)), an unmatched reference
    costs |t| and an unmatched output line costs |l|. The result is in [0, 1].
    """
    refs = [t for t in (line_text(x) for x in references) if t]
    if not refs:
        raise ValueError("Nonempty normalized references required")
    lines = [t for t in (line_text(x) for x in prediction.splitlines()) if t]
    n, m = len(refs), len(lines)
    costs, rows, cols = minimum_edit_assignment(refs, lines)
    errors = bound = pairs = exact = 0
    for i, j in zip(rows, cols, strict=True):
        errors += costs[i][j]
        if i < n and j < m:
            bound += max(len(refs[i]), len(lines[j]))
            pairs += 1
            exact += int(costs[i][j] == 0)
        elif i < n:
            bound += len(refs[i])
        elif j < m:
            bound += len(lines[j])
    return dict(
        references=n,
        output_lines=m,
        matched_pairs=pairs,
        exact_regions=exact,
        exact_recall=exact / n,
        errors=errors,
        bound=bound,
        ned=errors / bound,
    )
