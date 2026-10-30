"""Prepare OCR and QA records using the published train/dev split."""

import json
from collections import Counter
from pathlib import Path

from jasset.artifacts import digest, write_json
from jasset.data.release import (
    REPO,
    REVISION,
    TAG,
    extract_images,
    read_annotations,
    read_parquet_annotations,
)
from jasset.rewards import PROMPTS, PROTOCOL


def expand(row, root):
    iid = row["image_id"]
    common = dict(
        image_id=iid,
        source_scene_id=row["source_scene_id"],
        split=row["split"],
        source_image_sha256=row["image_sha256"],
        qa_category=row["qa_category"],
        source=dict(repository=REPO, revision=REVISION, image_id=iid),
        evidence=row["evidence"],
        protocol=PROTOCOL,
    )
    result = []
    for task in ("qa", "ocr"):
        image = f"images/{iid}.png"
        prompt = PROMPTS[task]
        if task == "ocr":
            planned = json.loads(row["planned_texts_json"])
            region_ids = [
                region["id"] for region in json.loads(row["scene_json"])["regions"]
            ]
            references = [planned[region_id] for region_id in region_ids]
        else:
            prompt += row["question"]
            references = list(dict.fromkeys([row["answer"], *row["acceptable_answers"]]))
        result.append(
            dict(
                common,
                id=f"{iid}:{task}",
                task=task,
                data_source=task,
                image=image,
                image_sha256=row["image_sha256"],
                prompt=prompt,
                references=references,
                reference_scope=(
                    "requested_planned_text"
                    if task == "ocr"
                    else "verified_answer"
                ),
                exhaustive_final_image_ocr=False,
            )
        )
    return result


def prepare(annotations, shards, out, limit_per_split=None):
    rows = (
        read_annotations(annotations)
        if annotations is not None
        else read_parquet_annotations(shards)
    )
    if limit_per_split is not None and limit_per_split <= 0:
        raise ValueError("Positive smoke limit required")
    selected = [
        r
        for split in ("train", "dev")
        for r in sorted(
            (r for r in rows if r["split"] == split), key=lambda r: r["image_id"]
        )[:limit_per_split]
    ]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources = extract_images(shards, rows, selected, out)
    records = [item for row in selected for item in expand(row, out)]
    for split, filename in (("train", "records.jsonl"), ("dev", "dev.jsonl")):
        (out / filename).write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n"
                for r in records
                if r["split"] == split
            )
        )
    write_json(out / "source-annotations.json", selected)
    files = {
        name: digest(out / name)
        for name in (
            "records.jsonl",
            "dev.jsonl",
            "source-annotations.json",
        )
    }
    manifest = dict(
        protocol=PROTOCOL,
        status="prepared",
        repository=REPO,
        tag=TAG,
        revision=REVISION,
        annotations_sha256=digest(annotations) if annotations is not None else None,
        input_shards_sha256=sources,
        smoke_limit_per_split=limit_per_split,
        ocr_reference_policy="requested",
        verified_final_image_ocr=False,
        selected_images=dict(Counter(r["split"] for r in selected)),
        task_counts=dict(Counter(f"{r['split']}/{r['task']}" for r in records)),
        outputs_sha256=files,
    )
    write_json(out / "manifest.json", manifest)
    return manifest


def validate_package(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["revision"] != REVISION or manifest["protocol"] != PROTOCOL:
        raise ValueError("Wrong FullGen release/protocol")
    for name, checksum in manifest["outputs_sha256"].items():
        if Path(name).name != name or digest(root / name) != checksum:
            raise ValueError("Modified package records")
    source_rows = json.loads((root / "source-annotations.json").read_text())
    expected_rows = [item for row in source_rows for item in expand(row, root)]
    rows = [
        json.loads(line)
        for name in ("records.jsonl", "dev.jsonl")
        for line in (root / name).read_text().splitlines()
    ]
    if rows != expected_rows:
        raise ValueError("Records differ from source annotation task expansion")
    for filename, split in (("records.jsonl", "train"), ("dev.jsonl", "dev")):
        if any(
            json.loads(line)["split"] != split
            for line in (root / filename).read_text().splitlines()
        ):
            raise ValueError("Wrong split file")
    ids, splits, checked = set(), {}, {}
    for row in rows:
        if row["id"] in ids or row["task"] != row["data_source"]:
            raise ValueError("Duplicate id or wrong routing")
        ids.add(row["id"])
        for key in (
            row["image_id"],
            row["source_scene_id"],
            row["source_image_sha256"],
        ):
            if key in splits and splits[key] != row["split"]:
                raise ValueError("Image family crosses splits")
            splits[key] = row["split"]
        path = Path(row["image"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe image path")
        if path not in checked:
            checked[path] = digest(root / path)
        if checked[path] != row["image_sha256"]:
            raise ValueError("Image bytes changed")
    return manifest
