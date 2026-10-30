"""Train the paper's OCR, QA, or joint GRPO condition."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
import yaml
from jasset.artifacts import read_jsonl
from jasset.data.prepare import validate_package
from jasset.rewards import qa_prompt, trl_reward


def select_records(records, tasks):
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - {"ocr", "qa"}:
        raise ValueError("Select unique OCR/QA tasks")
    images = {r["image_id"] for r in records}
    selected = [r for r in records if r["task"] in tasks]
    pairs = Counter((r["image_id"], r["task"]) for r in selected)
    if not images or any(pairs[i, t] != 1 for i in images for t in tasks):
        raise ValueError("Each task must cover every training image once")
    return selected


def training_rows(records, root):
    return [
        dict(
            prompt=[
                dict(
                    role="user",
                    content=[
                        dict(type="image"),
                        dict(
                            type="text",
                            text=qa_prompt(r["prompt"])
                            if r["task"] == "qa"
                            else r["prompt"],
                        ),
                    ],
                )
            ],
            images=[str((root / r["image"]).resolve())],
            reference_json=json.dumps(r["references"], ensure_ascii=False),
            image_id=r["id"],
            task=r["task"],
            evidence_json=json.dumps(
                [e["text"] for e in r["evidence"]], ensure_ascii=False
            ),
        )
        for r in records
    ]


def grpo_options(config, output, world_size):
    if world_size < 1 or 64 % world_size:
        raise ValueError("World size must divide 64 completions per generation round")
    return dict(
        output_dir=str(output),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64 // world_size,
        max_steps=-1,
        num_train_epochs=config["epochs"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=5e-7,
        max_grad_norm=1.0,
        weight_decay=0.0,
        seed=config["seed"],
        data_seed=config["seed"],
        logging_steps=1,
        save_steps=500,
        save_total_limit=None,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        num_generations=8,
        num_iterations=2,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_completion_length=4096,
        beta=0.02,
        epsilon=0.2,
        loss_type="grpo",
        scale_rewards="group",
        mask_truncated_completions=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=32768,
        chat_template_kwargs={"enable_thinking": False},
        model_init_kwargs={
            "revision": config["revision"],
            "dtype": "float32",
            "attn_implementation": "sdpa",
        },
        log_completions=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", help="Optional local copy of the pinned base model")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate full data and print configuration without GPU imports",
    )
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    config = yaml.safe_load(a.config.read_text())
    if config["revision"] != "15852e8c16360a2fea060d615a32b45270f8a8fc":
        raise ValueError("Paper training requires the pinned Qwen3.5-2B revision")
    validate_package(a.data)
    records = select_records(read_jsonl(a.data / "records.jsonl"), config["tasks"])
    if len({r["image_id"] for r in records}) != config["expected_images"]:
        raise ValueError("Training image count differs from configuration")
    world = int(os.environ.get("WORLD_SIZE", 4 if a.dry_run else 1))
    options = grpo_options(config, a.output, world)
    if a.dry_run:
        print(
            json.dumps(
                dict(records=len(records), config=config, grpo=options),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    import torch
    from datasets import Dataset, Image, Sequence
    from transformers import (
        AutoConfig,
        AutoProcessor,
        GenerationConfig,
        set_seed,
    )
    from trl import GRPOConfig, GRPOTrainer

    rank = int(os.environ.get("RANK", 0))
    if world > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
    from huggingface_hub import snapshot_download

    model = a.model or snapshot_download(config["model"], revision=config["revision"])
    a.output.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if a.resume:
        checkpoints = [
            path for path in a.output.glob("checkpoint-*")
            if path.name[11:].isdigit() and (path / "trainer_state.json").is_file()
        ]
        checkpoint = max(checkpoints, key=lambda path: int(path.name[11:]), default=None)
        if checkpoint is None:
            raise ValueError("No checkpoint found for --resume")
    elif any(a.output.iterdir()):
        raise ValueError("Use an empty output directory or --resume")
    if world > 1:
        torch.distributed.barrier()
    set_seed(config["seed"])
    processor = AutoProcessor.from_pretrained(
        model,
        revision=config["revision"],
        size={"shortest_edge": 65536, "longest_edge": 1605632},
    )
    processor.tokenizer.padding_side = "left"
    dataset = Dataset.from_list(training_rows(records, a.data)).cast_column(
        "images", Sequence(Image())
    )
    generation = GenerationConfig.from_model_config(
        AutoConfig.from_pretrained(model, revision=config["revision"])
    )
    if (Path(model) / "generation_config.json").exists():
        generation = GenerationConfig.from_pretrained(model)
    eos = generation.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {
        processor.tokenizer.eos_token_id
    }
    eos.discard(None)

    def reward(
        completions,
        completion_ids,
        reference_json,
        task,
        evidence_json,
        **kwargs,
    ):
        values = trl_reward(
            completions,
            completion_ids,
            reference_json,
            task,
            eos,
            evidence_json=evidence_json,
            format_weight=config["format_weight"],
            evidence_weight=config["evidence_weight"],
        )
        return [v["reward"] for v in values]

    args = GRPOConfig(**options)
    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        reward_funcs=reward,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    destination = a.output / "export"
    trainer.save_model(str(destination))
    trainer.accelerator.wait_for_everyone()
    if rank == 0:
        processor.save_pretrained(destination)


if __name__ == "__main__":
    main()
