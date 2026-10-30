"""Generate OCR/QA predictions using the paper's Qwen3.5 inference settings."""

import argparse
import json
from pathlib import Path
from jasset.artifacts import digest, write_json
from jasset.inference.extraction import final_answer, ocr_answer

TASKS = {
    "full-ocr": "jawildtext_ocr",
    "crop": "crop",
    "cc-ocr": "cc_ocr_ja",
    "mtvqa": "mtvqa_qa",
    "qa": "jawildtext_qa",
}


def validate_requests(root, requests, task):
    if not requests or len({r["key"] for r in requests}) != len(requests):
        raise ValueError("Requests must have nonempty, unique keys")
    limit = 512 if task == "crop" else 4096
    seen = {}
    for r in requests:
        if r["task"] != TASKS[task] or r["max_tokens"] != limit:
            raise ValueError("Request task or output limit differs from paper settings")
        if r.get("input_error"):
            continue
        image = Path(r["image"])
        if image.is_absolute() or ".." in image.parts:
            raise ValueError("Image path must be relative to --root")
        if image not in seen:
            seen[image] = digest(root / image)
        if seen[image] != r["image_sha256"]:
            raise ValueError("Image checksum mismatch")
    return limit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument(
        "--model", required=True, help="HF model ID or local final checkpoint/export"
    )
    p.add_argument("--revision", help="Required for remote models")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    rows = json.loads(a.requests.read_text())
    limit = validate_requests(a.root, rows, a.task)
    settings = dict(
        max_tokens=limit,
        max_model_len=4096 if a.task == "crop" else 32768,
        min_pixels=65536,
        max_pixels=1605632,
        temperature=0,
        seed=0,
        enable_thinking=False,
    )
    if not Path(a.model).is_dir() and not a.revision:
        p.error("--revision is required for remote models")
    if a.dry_run:
        print(json.dumps(dict(requests=len(rows), generation=settings), indent=2))
        return
    if a.output.exists():
        p.error("Output directory already exists")
    from transformers import AutoConfig, AutoProcessor, GenerationConfig
    from vllm import LLM, SamplingParams
    from PIL import Image

    processor = AutoProcessor.from_pretrained(a.model, revision=a.revision)
    try:
        generation = GenerationConfig.from_pretrained(a.model, revision=a.revision)
    except OSError:
        generation = GenerationConfig.from_model_config(
            AutoConfig.from_pretrained(a.model, revision=a.revision)
        )
    eos = generation.eos_token_id
    eos = [
        x
        for x in dict.fromkeys(
            [
                *(eos if isinstance(eos, list) else [eos]),
                processor.tokenizer.eos_token_id,
            ]
        )
        if x is not None
    ]
    llm = LLM(
        model=a.model,
        tokenizer=a.model,
        revision=a.revision,
        tokenizer_revision=a.revision,
        dtype="bfloat16",
        max_model_len=settings["max_model_len"],
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        seed=0,
        generation_config="vllm",
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 65536,
            "max_pixels": 1605632,
            "size": {"shortest_edge": 65536, "longest_edge": 1605632},
        },
    )
    a.output.mkdir(parents=True)
    write_json(
        a.output / "identity.json",
        dict(
            model=a.model,
            revision=a.revision,
            generation=settings,
            requests_sha256=digest(a.requests),
        ),
    )
    with (a.output / "predictions.jsonl").open("w") as stream:
        for start in range(0, len(rows), 8):
            batch = rows[start : start + 8]
            valid = [r for r in batch if not r.get("input_error")]
            inputs = []
            for r in valid:
                with Image.open(a.root / r["image"]) as image:
                    rgb = image.convert("RGB").copy()
                prompt = processor.apply_chat_template(
                    [
                        dict(
                            role="user",
                            content=[
                                dict(type="image"),
                                dict(type="text", text=r["prompt"]),
                            ],
                        )
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs.append(dict(prompt=prompt, multi_modal_data={"image": rgb}))
            generated = (
                llm.generate(
                    inputs,
                    SamplingParams(
                        temperature=0, max_tokens=limit, seed=0, stop_token_ids=eos
                    ),
                )
                if inputs
                else []
            )
            outputs = {
                r["key"]: g.outputs[0] for r, g in zip(valid, generated, strict=True)
            }
            for r in batch:
                result = {k: r[k] for k in ("key", "image_id", "image_sha256", "task")}
                if r.get("input_error"):
                    result.update(
                        status="input_error",
                        text=None,
                        capped=None,
                        input_error=r["input_error"],
                    )
                else:
                    g = outputs[r["key"]]
                    text, _ = final_answer(g.text, "")
                    if a.task in ("full-ocr", "crop", "cc-ocr"):
                        text, _ = ocr_answer(text)
                    result.update(
                        status="generated" if text is not None else "extraction_failed",
                        text=text,
                        raw_text=g.text,
                        capped=len(g.token_ids) >= limit,
                        token_ids=list(g.token_ids),
                    )
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()


if __name__ == "__main__":
    main()
