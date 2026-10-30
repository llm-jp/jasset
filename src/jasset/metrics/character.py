"""Character precision, recall and F1 (CC-OCR character bag, per image)."""

from collections import Counter

from jasset.metrics.normalize import ccocr_tokens

EPS = 1e-9  # CC-OCR ocr_evaluator.py


def character_scores(references, prediction):
    """References are the region strings of one image; their order is irrelevant."""
    gt = ccocr_tokens("\n".join(references))
    pred = ccocr_tokens(prediction)
    right = sum((Counter(gt) & Counter(pred)).values())
    recall, precision = right / (len(gt) + EPS), right / (len(pred) + EPS)
    return dict(
        reference_characters=len(gt),
        output_characters=len(pred),
        matched_characters=right,
        precision=precision,
        recall=recall,
        f1=2 * recall * precision / (recall + precision + EPS),
    )


def micro_f1(rows):
    """Pooled F1 over images (CC-OCR ``mirco_f1_score``); auxiliary only."""
    right = sum(r["matched_characters"] for r in rows)
    recall = right / (sum(r["reference_characters"] for r in rows) + EPS)
    precision = right / (sum(r["output_characters"] for r in rows) + EPS)
    return 2 * recall * precision / (recall + precision + EPS)
