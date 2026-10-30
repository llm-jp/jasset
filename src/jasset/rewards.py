"""OCR set reward and format-gated answer/evidence reward used in the paper."""

import json
import math
import re
from collections import Counter
from jasset.metrics.normalize import normalize
from jasset.metrics.qa import extract_boxed
from jasset.metrics.sets import normalized_set_scores
from jasset.prompts import OCR_PROMPT

PROTOCOL = "jasset-ocr-qa-v1"
PROMPTS = {
    "ocr": OCR_PROMPT,
    "qa": "画像を見て質問に答えてください。説明や根拠を加えず、短い回答のみを出力してください。\n質問: ",
}
QA_REASONING_PROMPT = (
    "画像を参照して回答してください。推論過程を出力し、"
    "質問に答えるのに必要な画像内の文字列を「」で引用しながら、"
    "それらを根拠に回答を導いてください。"
    "最終回答は必ず \\boxed{...} で囲み、ボックス内には最終回答のみを1つだけ記載してください。"
)


def qa_prompt(original):
    if not original.startswith(PROMPTS["qa"]):
        raise ValueError("Unexpected QA prompt")
    return original[len(PROMPTS["qa"]) :] + "\n" + QA_REASONING_PROMPT


def ocr_reward(prediction, references, capped=False):
    refs = [normalize(x) for x in references]
    preds = [normalize(x) for x in prediction.splitlines() if normalize(x)]
    scores = normalized_set_scores(refs, preds)
    return dict(
        reward=0.0 if capped else (scores["exact_f1"] + scores["soft_f1"]) / 2,
        **scores,
        capped=capped,
    )


def score(
    text,
    task,
    references,
    capped=False,
    *,
    evidence_references=(),
    format_weight=0.2,
    evidence_weight=0.2,
):
    if (
        task not in PROMPTS
        or not references
        or any(not normalize(x) for x in references)
    ):
        raise ValueError("Unknown task or empty references")
    if not isinstance(text, str):
        raise ValueError("Prediction must be text")
    if any(not math.isfinite(w) or w < 0 for w in (format_weight, evidence_weight)):
        raise ValueError("Reward weights must be finite and nonnegative")
    result = dict(capped=capped, output_characters=len(normalize(text)))
    if task == "ocr":
        return result | ocr_reward(text, references, capped)
    boxes = len(re.findall(r"\\boxed(?![A-Za-z])", text))
    answer = extract_boxed(text) if boxes == 1 else None
    valid = answer is not None and bool(normalize(answer))
    gold = list(
        dict.fromkeys(normalize(s) for s in evidence_references if normalize(s))
    )
    if not gold:
        raise ValueError("QA requires verified evidence references")
    quotes = [normalize(s) for s in re.findall("「([^」]*)」", text)]
    matches = sum((Counter(quotes) & Counter(gold)).values())
    evidence = 2 * matches / (len(quotes) + len(gold))
    correct = float(valid and normalize(answer) in {normalize(s) for s in references})
    gate = valid and not capped
    result.update(
        answer=answer,
        answer_correct=correct,
        format_valid=valid,
        evidence_f1=evidence,
        format_reward=format_weight if gate else 0.0,
        answer_reward=correct if gate else 0.0,
        evidence_reward=evidence_weight * evidence if gate else 0.0,
    )
    result["reward"] = sum(
        result[k] for k in ("format_reward", "answer_reward", "evidence_reward")
    )
    return result


def trl_reward(
    completions,
    completion_ids,
    reference_json,
    task,
    eos_ids,
    *,
    evidence_json,
    format_weight=0.2,
    evidence_weight=0.2,
):
    return [
        score(
            c[0]["content"] if isinstance(c, list) else c,
            t,
            json.loads(r),
            not ids or ids[-1] not in eos_ids,
            evidence_references=json.loads(e),
            format_weight=format_weight,
            evidence_weight=evidence_weight,
        )
        for c, ids, r, t, e in zip(
            completions,
            completion_ids,
            reference_json,
            task,
            evidence_json,
            strict=True,
        )
    ]
