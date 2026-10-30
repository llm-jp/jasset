"""Prepare the pinned JaSSeT release for OCR/QA training."""

import argparse
import json
from pathlib import Path
from jasset.data.prepare import prepare
from jasset.data.release import REPO, REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--release",
        type=Path,
        help="Local HF snapshot containing Parquet shards",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--limit-per-split",
        type=int,
        help="Small preparation check; omit for paper training",
    )
    a = p.parse_args()
    if a.release is None:
        from huggingface_hub import snapshot_download

        a.release = Path(
            snapshot_download(
                repo_id=REPO,
                repo_type="dataset",
                revision=REVISION,
                allow_patterns=[
                    "*.parquet",
                    "**/*.parquet",
                ],
            )
        )
    result = prepare(
        None, list(a.release.rglob("*.parquet")), a.output, a.limit_per_split
    )
    print(json.dumps(result["task_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
