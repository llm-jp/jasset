"""Package the exact archived crop pixels and input errors for crop OCR evaluation."""

import argparse
import json
from pathlib import Path

from jasset.artifacts import digest, write_json
from jasset.protocol import CROP_GENERATION, CROP_PROMPT

INPUTS_SHA256 = "9f865748253b4b0200f20a5fd1e0a284e0a613de6af9e3e05a9f3d1132f58b85"
GOLD_SHA256 = "5d069997d373e7abce33917a3ebd72e0be30e5e241ab9401ade39188968de2a6"


def prepare(source, output):
    if output.exists():
        raise ValueError("Fresh crop package required")
    for name, expected in (("inputs.json", INPUTS_SHA256), ("gold.json", GOLD_SHA256)):
        if digest(source / name) != expected:
            raise ValueError("Archived crop input or gold changed")
    inputs = json.loads((source / "inputs.json").read_text())
    gold = json.loads((source / "gold.json").read_text())
    if len(inputs) != 33608 or len({r["id"] for r in inputs}) != 33608:
        raise ValueError("Wrong crop coverage")
    requests, images = [], {}
    for row, reference in zip(inputs, gold, strict=True):
        if row["id"] != reference["id"] or row["sha256"] != reference["image_sha256"]:
            raise ValueError("Crop reference identity differs")
        image = None
        if not row["input_error"]:
            path = source / row["image"]
            if digest(path) != row["sha256"]:
                raise ValueError("Archived crop pixels changed")
            image = "images/" + row["sha256"] + path.suffix
            images[image] = str(path.resolve())
        requests.append(
            dict(
                key=row["id"],
                image_id=row["source_id"],
                task="crop",
                image=image,
                image_sha256=row["sha256"],
                input_error=row["input_error"],
                prompt=CROP_PROMPT,
                max_tokens=CROP_GENERATION["max_tokens"],
            )
        )
    if sum(bool(r["input_error"]) for r in requests) != 17:
        raise ValueError("Archived input-error count differs")
    (output / "images").mkdir(parents=True)
    for relative, path in images.items():
        (output / relative).symlink_to(path)
    write_json(output / "crop-requests.json", requests)
    write_json(output / "crop-gold.json", gold)
    write_json(
        output / "receipt.json",
        dict(
            task_family="crop",
            rows=len(requests),
            input_errors=17,
            unique_images=len(images),
            source_hashes={
                str(source / "inputs.json"): INPUTS_SHA256,
                str(source / "gold.json"): GOLD_SHA256,
            },
            files={p.name: digest(p) for p in output.glob("*.json")},
            generation=CROP_GENERATION,
            transfer="Dereference symlinks with rsync -aL; runner validates every image hash",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
