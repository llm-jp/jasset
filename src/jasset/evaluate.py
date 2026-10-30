"""Score saved predictions with the paper's fixed evaluation denominators."""

import argparse
import importlib
import json
from pathlib import Path
from jasset.artifacts import digest, read_jsonl, write_json
from jasset.evaluation.scoring import score, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task", choices=["full-ocr", "crop", "cc-ocr", "mtvqa", "qa"], required=True
    )
    for name in ["gold", "predictions", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--cohort", type=Path, help="Frozen JaWildText primary-image cohort JSON"
    )
    p.add_argument("--requests", type=Path, help="QA requests JSON used for inference")
    p.add_argument(
        "--cache", type=Path, help="QA verdict cache directory"
    )
    p.add_argument("--reuse-cache", type=Path)
    p.add_argument("--judge", help="User callable as module:function; accepts messages, returns text")
    p.add_argument("--judge-id", help="Cache identity for the model, revision and generation settings")
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    if bool(a.judge) != bool(a.judge_id):
        p.error("Specify --judge and --judge-id together")
    if a.judge and a.task != "qa":
        p.error("--judge is only supported for QA")
    if a.output.exists():
        p.error("Output directory already exists")
    gold, predictions = json.loads(a.gold.read_text()), read_jsonl(a.predictions)
    if a.task == "qa":
        if a.requests is None or a.cache is None:
            p.error("QA requires --requests and --cache")
        import fcntl
        from jasset.evaluation.judge import Cache, score as judge_score

        judge = None
        if a.judge:
            module, separator, name = a.judge.partition(":")
            if not separator or not module or not name:
                p.error("--judge must have the form module:function")
            judge = getattr(importlib.import_module(module), name)
            if not callable(judge):
                p.error("--judge must refer to a callable")
        cache = Cache(a.cache, a.reuse_cache or a.cache / "cache", judge, a.judge_id)
        with (a.cache / "scoring.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            rows = judge_score(
                json.loads(a.requests.read_text()), gold, predictions, cache
            )
        complete = [r for r in rows if r["status"] == "complete"]
        n = len(rows)
        if not n:
            p.error("Empty QA set")
        summary = dict(
            expected=n,
            complete=len(complete),
            capped=sum(r["capped"] is True for r in rows),
            content_accuracy=sum(r["metrics"]["content"] for r in complete) / n
            if len(complete) == n
            else None,
            format_accuracy=sum(r["metrics"]["boxed"] for r in complete) / n
            if len(complete) == n
            else None,
            judge_calls=cache.calls,
            judge_id=a.judge_id,
            cache_hits=cache.hits,
        )
    else:
        cohort = json.loads(a.cohort.read_text()) if a.cohort else None
        rows = score(gold, predictions, a.task, cohort, a.workers)
        summary = report(rows, a.task)
    expected = {
        "full-ocr": 580,
        "crop": 32409,
        "cc-ocr": 150,
        "mtvqa": 828,
        "qa": 1025,
    }[a.task]
    if len(rows) != expected:
        raise ValueError(
            f"Paper {a.task} requires {expected} records; received {len(rows)}"
        )
    a.output.mkdir(parents=True)
    (a.output / "rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    write_json(a.output / "summary.json", summary)
    write_json(
        a.output / "receipt.json",
        {
            "task": a.task,
            "sources": {
                str(x): digest(x)
                for x in (a.gold, a.predictions, a.cohort, a.requests)
                if x is not None
            },
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
