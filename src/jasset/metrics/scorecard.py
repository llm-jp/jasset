"""Per-image rows and summaries for full-image OCR (the single scoring entry point)."""

import statistics
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from jasset.metrics import VERSION
from jasset.metrics.cer import order_free_cer
from jasset.metrics.character import character_scores, micro_f1
from jasset.metrics.ned import ned_scores

MAIN = ("precision", "recall", "f1", "ned", "cer")


def score_image(
    references, prediction, capped=False, max_iterations=100, full_cer=False
):
    """``full_cer`` also resolves the unclipped CER (slow on long repetitive outputs)."""
    if not isinstance(references, list) or any(
        not isinstance(t, str) for t in references
    ):
        raise ValueError("A list of region reference strings is required")
    if not isinstance(prediction, str):
        raise ValueError("Prediction text required; use '' for an unavailable output")
    return dict(
        capped=bool(capped),
        empty=not prediction.strip(),
        character=character_scores(references, prediction),
        ned=ned_scores(references, prediction),
        cer=order_free_cer(
            references, prediction, max_iterations=max_iterations, clip=not full_cer
        ),
    )


def _score_item(item):
    key, image_id, references, text, capped, max_iterations, full_cer = item
    return dict(
        key=key,
        image_id=image_id,
        **score_image(references, text, capped, max_iterations, full_cer),
    )


def score_images(items, max_iterations=100, workers=1, full_cer=False):
    """``items`` are (key, image_id, references, text, capped); order is preserved."""
    jobs = [(*item, max_iterations, full_cer) for item in items]
    if workers <= 1:
        return [_score_item(job) for job in jobs]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_score_item, jobs, chunksize=1))


def aggregate(rows):
    """Main values are image means; per-image CER is clipped at 1."""
    return dict(
        precision=float(np.mean([r["character"]["precision"] for r in rows])),
        recall=float(np.mean([r["character"]["recall"] for r in rows])),
        f1=float(np.mean([r["character"]["f1"] for r in rows])),
        ned=float(np.mean([r["ned"]["ned"] for r in rows])),
        cer=float(np.mean([r["cer"]["cer_clipped"] for r in rows])),
    )


def summarize(rows):
    if not rows:
        raise ValueError("No rows")
    cer = [r["cer"] for r in rows]
    full = not any(x["stopped"] for x in cer)
    N = sum(x["N"] for x in cer)
    # Additive diagnostic; old saved score rows can still be summarized.
    exact = (
        sum(r["ned"]["exact_regions"] for r in rows)
        if all("exact_regions" in r["ned"] for r in rows)
        else None
    )
    return dict(
        metrics_version=VERSION,
        images=len(rows),
        capped=sum(r["capped"] for r in rows),
        empty=sum(r["empty"] for r in rows),
        **aggregate(rows),
        character_micro_f1=micro_f1([r["character"] for r in rows]),
        exact_region_recall=(
            exact / sum(r["ned"]["references"] for r in rows)
            if exact is not None
            else None
        ),
        cer_detail=dict(
            clipped_images=sum(x["cer_clipped"] >= 1.0 for x in cer),
            exact_images=sum(x["clipped_exact"] for x in cer),
            exact_rate=sum(x["clipped_exact"] for x in cer) / len(cer),
            mean_gap=float(np.mean([x["clipped_gap"] for x in cer])),
            median_image_cer=statistics.median(x["cer_clipped"] for x in cer),
            reference_characters=N,
            pooled_cer=sum(x["E_ub"] for x in cer) / N if full else None,
            substitutions=sum(x["S"] for x in cer) if full else None,
            deletions=sum(x["D"] for x in cer) if full else None,
            insertions=sum(x["I"] for x in cer) if full else None,
        ),
    )
