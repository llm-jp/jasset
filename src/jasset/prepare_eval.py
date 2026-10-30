"""Freeze real-image evaluation requests separately from gold annotations."""

import argparse
import json
from pathlib import Path

from jasset.artifacts import digest, write_json
from jasset.protocol import GENERATION
from jasset.prompts import OCR_PROMPT, task_prompt

CC_REVISION = "c64517e92179991d509776064174776700cdd5a2"


def prepare(prepared, cc_manifest, output):
    prepared, cc_manifest, output = map(Path, (prepared, cc_manifest, output))
    if output.exists():
        raise ValueError("Fresh evaluation package required")
    sources = dict(
        jawildtext=prepared / "jawildtext_test.jsonl",
        mtvqa=prepared / "mtvqa_test.jsonl",
        cc_ocr=cc_manifest,
    )
    rows = {
        key: [json.loads(s) for s in path.read_text().splitlines()]
        for key, path in sources.items()
    }
    if {k: len(v) for k, v in rows.items()} != dict(
        jawildtext=1025, mtvqa=828, cc_ocr=150
    ):
        raise ValueError("Unexpected evaluation denominators")
    if any(
        r["source_revision"] != CC_REVISION or r["subset"] != "Japanese"
        for r in rows["cc_ocr"]
    ):
        raise ValueError("Wrong CC-OCR source")
    requests, gold, images, image_hashes = {}, {}, {}, {}

    def add(group, row, key, prompt, reference):
        image = Path(row.get("image_path") or row["image"]).resolve()
        if image not in image_hashes:
            image_hashes[image] = digest(image)
        sha = image_hashes[image]
        if row.get("image_sha256", sha) != sha:
            raise ValueError("Changed evaluation image")
        relative = "images/" + sha + image.suffix.lower()
        images[relative] = dict(source=str(image), sha256=sha)
        requests.setdefault(group, []).append(
            dict(
                key=key,
                image_id=row["image_id"],
                task=group,
                image=relative,
                image_sha256=sha,
                prompt=prompt,
                max_tokens=GENERATION["max_tokens"],
            )
        )
        gold.setdefault(group, []).append(
            dict(
                key=key,
                image_id=row["image_id"],
                reference=reference,
                reference_unit="whole_transcript"
                if group == "cc_ocr_ja"
                else "region_strings"
                if group in ("jawildtext_ocr", "jawildtext_evidence")
                else "answer",
            )
        )

    seen_images = {}
    for row in rows["jawildtext"]:
        reference = [r["text"] for r in row["regions"]]
        if row["image_id"] not in seen_images:
            add("jawildtext_ocr", row, row["image_id"], OCR_PROMPT, reference)
            seen_images[row["image_id"]] = reference
        elif seen_images[row["image_id"]] != reference:
            raise ValueError("Inconsistent image annotations")
        add("jawildtext_qa", row, row["id"], task_prompt("qa", row), row["answer"])

    if len(seen_images) != 745:
        raise ValueError("Expected 745 JaWildText images")
    for row in rows["mtvqa"]:
        add("mtvqa_qa", row, row["id"], task_prompt("qa", row), row["answer"])
    for row in rows["cc_ocr"]:
        add("cc_ocr_ja", row, row["id"], OCR_PROMPT, row["answer"])
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for relative, value in images.items():
        (output / relative).symlink_to(value["source"])
    for group in requests:
        keys = [r["key"] for r in requests[group]]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate request keys")
        write_json(output / f"{group}-requests.json", requests[group])
        write_json(output / f"{group}-gold.json", gold[group])
    receipt = dict(
        source_files={
            k: dict(path=str(p), sha256=digest(p)) for k, p in sources.items()
        },
        groups={k: len(v) for k, v in requests.items()},
        images=images,
        files={p.name: digest(p) for p in output.glob("*.json")},
        generation=GENERATION,
        cc_source=f"https://huggingface.co/datasets/wulipc/CC-OCR/tree/{CC_REVISION}",
        scope="Full-image OCR, QA and holdouts. Crop uses the existing frozen crop inputs.",
        transfer=(
            "Dereference image symlinks (rsync -aL); verify hashes on destination before inference."
        ),
    )
    write_json(output / "receipt.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "cc-manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = prepare(a.prepared, a.cc_manifest, a.output)
    print(json.dumps(dict(groups=result["groups"], images=len(result["images"]))))


if __name__ == "__main__":
    main()
