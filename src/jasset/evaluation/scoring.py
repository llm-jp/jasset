"""Paper metrics with explicit fixed and completed denominators for each task."""

from collections import Counter
from jasset.evaluation.records import TASKS, index, primary_gold
from jasset.metrics import VERSION
from jasset.metrics.character import character_scores, micro_f1
from jasset.metrics.qa import MTVQA_REVISION, mtvqa_score
from jasset.metrics.region import region_scores


def references(gold, task, cohort):
    if task == "crop":
        if cohort is None:
            raise ValueError("Frozen cohort required for crop subsets")
        primary = {r["id"] for r in cohort if r["primary"]}
        all_refs = index(
            [
                dict(
                    r,
                    key=r["id"],
                    image_id=r["source_id"],
                    reference=r["text"],
                    common580=r["source_id"] in primary,
                )
                for r in gold
            ]
        )
        refs = {k: r for k, r in all_refs.items() if r["primary"]}
    elif task == "full-ocr":
        if cohort is None:
            raise ValueError("Frozen cohort required for full-image OCR")
        all_refs = index(gold)
        refs = primary_gold(gold, cohort)
    else:
        all_refs = refs = index(gold)
        expected_unit = "whole_transcript" if task == "cc-ocr" else "answer"
        if any((r["reference_unit"] != expected_unit for r in refs.values())):
            raise ValueError("Wrong reference unit")
    if not refs:
        raise ValueError("Empty evaluation denominator")
    return (all_refs, refs)


def score(gold, predictions, task, cohort=None, workers=1, iterations=100):
    if task not in TASKS:
        raise ValueError("Unknown paper task")
    all_refs, refs = references(gold, task, cohort)
    preds = index(predictions)
    if not preds.keys() <= all_refs.keys():
        raise ValueError("Predictions outside frozen gold")
    rows, full_items = ([], [])
    for key, ref in refs.items():
        pred = preds.get(key)
        row = dict(key=key, image_id=ref["image_id"], status="missing", capped=None)
        if task == "crop":
            row.update(common580=ref["common580"], size_bin=ref["size_bin"])
        text = ""
        if pred is not None:
            if pred["image_id"] != ref["image_id"]:
                raise ValueError("Image identity changed")
            if "task" in pred and pred["task"] != TASKS[task]:
                raise ValueError("Prediction task changed")
            if task in ("crop",) and pred.get("image_sha256") != ref["image_sha256"]:
                raise ValueError("Input pixels changed")
            status = pred.get("status", "generated")
            if status == "generated":
                if not isinstance(pred.get("text"), str) or not isinstance(
                    pred.get("capped"), bool
                ):
                    raise ValueError(
                        "Generated output requires text and explicit cap status"
                    )
                text = pred["text"]
            row.update(status=status, capped=pred.get("capped"))
        if task == "crop":
            row["metrics"] = region_scores(ref["reference"], text)
        elif task == "cc-ocr":
            row["metrics"] = dict(
                character=character_scores([ref["reference"]], text),
                ned=region_scores(ref["reference"], text)["ned"],
            )
        elif task == "mtvqa":
            row["metrics"] = dict(
                correct=mtvqa_score(text, ref["reference"])
                if row["status"] == "generated"
                else 0.0
            )
        else:
            full_items.append(
                (key, ref["image_id"], ref["reference"], text, row["capped"] is True)
            )
        rows.append(row)
    if task == "full-ocr":
        from jasset.metrics.scorecard import score_images

        for row, metrics in zip(
            rows, score_images(full_items, iterations, workers), strict=True
        ):
            row["metrics"] = metrics
    return rows


def aggregate(rows, task):
    if not rows:
        return None
    n = len(rows)
    if task == "full-ocr":
        from jasset.metrics.scorecard import summarize as summarize_full

        return summarize_full([r["metrics"] for r in rows])
    if task == "crop":
        return dict(
            ned=sum((r["metrics"]["ned"] for r in rows)) / n,
            em=sum((r["metrics"]["em"] for r in rows)) / n,
        )
    if task in ("cc-ocr",):
        chars = [r["metrics"]["character"] for r in rows]
        result = dict(
            ned=sum((r["metrics"]["ned"] for r in rows)) / n,
            character_macro_f1=sum((r["f1"] for r in chars)) / n,
            character_micro_f1=micro_f1(chars),
        )
        return result
    return dict(accuracy=sum((r["metrics"]["correct"] for r in rows)) / n)


def summarize(rows, task):
    if not rows:
        raise ValueError("Empty evaluation denominator")
    complete = [r for r in rows if r["status"] == "generated"]
    return dict(
        expected=len(rows),
        generated=len(complete),
        status_counts=dict(Counter((r["status"] for r in rows))),
        capped=sum((r["capped"] is True for r in rows)),
        cap_unknown=sum((r["capped"] is None for r in rows)),
        fixed=aggregate(rows, task),
        completed=aggregate(complete, task),
    )


def report(rows, task):
    result = dict(
        schema=1,
        task=task,
        metrics_version=VERSION,
        definitions=dict(
            fixed="All references; unavailable outputs receive empty text (QA incorrect)",
            completed="Generated outputs only, including empty and capped generations",
            units="Rates in [0,1]; multiply by 100 only when displaying percentages",
        ),
        **summarize(rows, task),
    )
    if task == "crop":
        groups = {"common580": [r for r in rows if r["common580"]]}
        groups.update(
            {
                f"size:{b}": [r for r in rows if r["size_bin"] == b]
                for b in sorted({r["size_bin"] for r in rows})
            }
        )
        result["subsets"] = {
            name: summarize(group, task) for name, group in groups.items() if group
        }
    if task == "mtvqa":
        result["official_revision"] = MTVQA_REVISION
        result["official_rule"] = (
            "Case-insensitive annotation substring; remove periods in prediction"
        )
    return result
