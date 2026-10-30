"""Read the published annotations and extract training images."""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from jasset.artifacts import digest

REPO = "llm-jp/jasset"
TAG = "sceneqa-fullgen-1.0.0-4k-20261003-r1"
REVISION = "567bad6b19f1d1ff7a6ba257523c554542ffa47b"
ANNOTATIONS_SHA256 = "07b066736e31e5779156a74bd59b5fe6fdbd9c97617e296210af0f8db1e3b7f0"
COUNTS = {"train": 3500, "dev": 500}


def read_annotations(path):
    if digest(path) != ANNOTATIONS_SHA256:
        raise ValueError("Annotations differ from pinned public release")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if dict(Counter(row["split"] for row in rows)) != COUNTS:
        raise ValueError("Expected frozen train 3500 / dev 500")
    return rows


def read_parquet_annotations(shards):
    """Read annotations without loading the embedded image column."""
    import pyarrow.parquet as pq

    rows = []
    for shard in sorted(shards):
        parquet = pq.ParquetFile(shard)
        columns = [name for name in parquet.schema_arrow.names if name != "image"]
        rows.extend(parquet.read(columns=columns).to_pylist())
    if dict(Counter(row["split"] for row in rows)) != COUNTS:
        raise ValueError("Expected frozen train 3500 / dev 500")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image IDs in release")
    canonical = json.dumps(
        sorted(rows, key=lambda row: row["image_id"]),
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != "0728a112961626c90f770ddf8235732ed3c5297807d8aba1723a362f1ea7ce3c":
        raise ValueError("Annotations differ from pinned public release")
    return rows


def extract_images(shards, annotations, selected, out):
    """Bind original PNG bytes to the pinned annotations."""
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    wanted = {row["image_id"]: row for row in selected}
    all_rows = {row["image_id"]: row for row in annotations}
    found, sources = set(), {}
    (out / "images").mkdir(parents=True)
    for shard in sorted(shards):
        sources[str(shard)] = digest(shard)
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=1):
            for row in batch.to_pylist():
                iid = row["image_id"]
                if iid not in wanted:
                    continue
                if iid in found:
                    raise ValueError("Duplicate selected image in shards")
                source = all_rows[iid]
                for key, value in row.items():
                    if key != "image" and value != source[key]:
                        raise ValueError(f"Parquet/sidecar mismatch: {iid}/{key}")
                raw = row["image"]["bytes"]
                if hashlib.sha256(raw).hexdigest() != source["image_sha256"]:
                    raise ValueError("Image bytes differ from release")
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    if image.size != (source["width"], source["height"]):
                        raise ValueError("Image dimensions disagree")
                (out / "images" / f"{iid}.png").write_bytes(raw)
                found.add(iid)
    if found != set(wanted):
        raise ValueError(
            f"Missing {len(set(wanted) - found)} selected images; supply their shards"
        )
    return sources
