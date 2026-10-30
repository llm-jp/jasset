"""Score QA using cached judgments or a user-supplied model callable."""

import hashlib
import json
from pathlib import Path

from jasset.artifacts import digest
from jasset.prompts import GRADER_TEMPLATE
from jasset.metrics.qa import extract_boxed, parse_verdict
from jasset.artifacts import write_json as atomic_json


def fingerprint(request):
    return hashlib.sha256(
        json.dumps(request, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


class Cache:
    def __init__(self, root, reuse, judge=None, judge_id=None):
        if judge is not None and not judge_id:
            raise ValueError("A judge_id is required for new judgments")
        self.root, self.reuse = Path(root), Path(reuse)
        self.judge_id = judge_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.calls = self.hits = 0
        self.judge = judge
        self.sources = {}

    def evaluate(self, request):
        key = fingerprint(request)
        path = self.root / "cache" / (key + ".json")
        source = path if path.exists() else self.reuse / path.name
        prior = (
            json.loads(source.read_text())
            if source.exists()
            else dict(
                request=request,
                attempts=[],
                score=None,
            )
        )
        if prior["request"] != request:
            raise ValueError("QA cache request drift")
        if self.judge is not None and prior.get("judge_id") != self.judge_id:
            prior = dict(request=request, attempts=[], score=None, judge_id=self.judge_id)
        if prior["score"] is not None:
            if parse_verdict(prior["attempts"][-1]["content"]) != prior["score"]:
                raise ValueError("QA cache verdict drift")
            self.hits += 1
            self.sources[str(source)] = digest(source)
            return prior["score"]
        if self.judge is None:
            return None
        messages = [
            dict(role="system", content="あなたは回答の正誤を判定する専門家です。"),
            dict(role="user", content=GRADER_TEMPLATE.format(
                question=request["question"], gold_answer=request["gold"],
                prediction=request["prediction"],
            )),
        ]
        self.calls += 1
        content = self.judge(messages)
        if not isinstance(content, str):
            raise TypeError("Judge callable must return response text (str)")
        prior["attempts"].append(dict(content=content))
        prior["score"] = parse_verdict(content)
        atomic_json(path, prior)
        saved = path if path.exists() else source
        self.sources[str(saved)] = digest(saved)
        return prior["score"]


def score(requests, gold, predictions, cache):
    def index(rows):
        result = {r["key"]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate QA keys")
        return result

    req, refs, predicted = index(requests), index(gold), index(predictions)
    if req.keys() != refs.keys() or not predicted.keys() <= req.keys():
        raise ValueError("QA coverage mismatch")
    result = []
    for key, row in req.items():
        output = dict(key=key, image_id=row["image_id"], status="missing", capped=None)
        pred = predicted.get(key)
        if pred is None:
            result.append(output)
            continue
        if (
            row["image_id"] != pred["image_id"]
            or row["image_id"] != refs[key]["image_id"]
        ):
            raise ValueError("QA image identity drift")
        output.update(status=pred["status"], capped=pred["capped"])
        if pred["status"] != "generated":
            result.append(output)
            continue
        final = pred["text"]
        if not isinstance(final, str):
            raise ValueError("Generated QA requires text")
        boxed = extract_boxed(final)
        metrics = dict(format_valid=int(bool(boxed)))
        for mode, answer in (("boxed", boxed), ("content", boxed or final)):
            metrics[mode] = (
                cache.evaluate(
                    dict(
                        question=row["prompt"],
                        gold=refs[key]["reference"],
                        prediction=answer,
                    )
                )
                if answer
                else 0.0
            )
        output.update(
            status="complete"
            if all(v is not None for v in metrics.values())
            else "unscored",
            boxed_answer=boxed,
            metrics=metrics,
        )
        result.append(output)
    return result
