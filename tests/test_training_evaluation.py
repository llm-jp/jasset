import json
from pathlib import Path
import pytest
from jasset.rewards import score, qa_prompt, PROMPTS
from jasset.evaluation.scoring import score as evaluate, report
from jasset.train import grpo_options, select_records
from jasset.predict import validate_requests

ROOT = Path(__file__).parent


@pytest.mark.parametrize("case", json.loads((ROOT / "reward_cases.json").read_text()))
def test_reward_against_recorded_implementation(case):
    result = score(
        case["text"],
        case["task"],
        case["references"],
        case["capped"],
        evidence_references=["東京"],
        evidence_weight=case["evidence_weight"],
    )
    assert result["reward"] == case["reward"]


@pytest.mark.parametrize(
    "case", json.loads((ROOT / "evaluation_cases.json").read_text())
)
def test_metrics_against_recorded_implementation(case):
    rows = evaluate(case["gold"], case["predictions"], case["task"], case["cohort"])
    assert report(rows, case["task"]) == case["expected"]


def test_qa_training_prompt_is_not_short_answer_prompt():
    text = qa_prompt(PROMPTS["qa"] + "駅名は？")
    assert text.startswith(
        "駅名は？\n画像を参照して回答してください。推論過程を出力し、"
    )
    assert "「」で引用" in text
    assert "\\boxed{...}" in text


def test_task_selection_preserves_order_and_detects_missing_task():
    rows = [dict(image_id=str(i), task=t) for i in range(2) for t in ("qa", "ocr")]
    assert select_records(rows, ["ocr", "qa"]) == rows
    assert [r["image_id"] for r in select_records(rows, ["ocr"])] == ["0", "1"]
    with pytest.raises(ValueError):
        select_records(rows[:-1], ["ocr"])


def test_training_round_and_precision():
    options = grpo_options(dict(epochs=1, seed=99, revision="pinned"), "out", 4)
    assert options["gradient_accumulation_steps"] == 16
    assert options["num_generations"] == 8
    assert options["num_iterations"] == 2
    assert options["model_init_kwargs"]["dtype"] == "float32"
    assert options["bf16"] is True
    assert options["mask_truncated_completions"] is True
    with pytest.raises(ValueError):
        grpo_options(dict(epochs=1, seed=99, revision="pinned"), "out", 3)


def test_prediction_package_checks_image_identity(tmp_path):
    from jasset.artifacts import digest

    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    row = dict(
        key="a",
        task="jawildtext_ocr",
        max_tokens=4096,
        image="image.png",
        image_sha256=digest(image),
    )
    assert validate_requests(tmp_path, [row], "full-ocr") == 4096
    image.write_bytes(b"changed")
    with pytest.raises(ValueError):
        validate_requests(tmp_path, [row], "full-ocr")


def test_missing_prediction_keeps_fixed_denominator():
    gold = [
        dict(key="a", image_id="a", reference=["東京"], reference_unit="answer"),
        dict(key="b", image_id="b", reference=["大阪"], reference_unit="answer"),
    ]
    predictions = [dict(key="a", image_id="a", text="東京", capped=False)]
    result = report(evaluate(gold, predictions, "mtvqa"), "mtvqa")
    assert result["fixed"]["accuracy"] == 0.5
    assert result["completed"]["accuracy"] == 1
    assert result["generated"] == 1


def test_duplicate_prediction_is_rejected():
    gold = [dict(key="a", image_id="a", reference="東京", reference_unit="answer")]
    row = dict(key="a", image_id="a", text="東京", capped=False)
    with pytest.raises(ValueError):
        evaluate(gold, [row, row], "mtvqa")


def test_qa_judging_preserves_content_and_format_scores():
    from jasset.evaluation.judge import score as judge

    class Cache:
        def evaluate(self, request):
            return float(request["prediction"] == "東京")

    requests = [dict(key="a", image_id="i", prompt="駅名は？")]
    gold = [dict(key="a", image_id="i", reference="東京")]
    predictions = [
        dict(key="a", image_id="i", text="東京", status="generated", capped=False)
    ]
    row = judge(requests, gold, predictions, Cache())[0]
    assert row["metrics"]["content"] == 1
    assert row["metrics"]["boxed"] == 0
    predictions[0]["text"] = r"\boxed{東京}"
    row = judge(requests, gold, predictions, Cache())[0]
    assert row["metrics"]["content"] == row["metrics"]["boxed"] == 1


def test_qa_cache_miss_remains_unscored():
    from jasset.evaluation.judge import score as judge

    class Cache:
        def evaluate(self, request):
            return None

    row = judge(
        [dict(key="a", image_id="i", prompt="駅名は？")],
        [dict(key="a", image_id="i", reference="東京")],
        [
            dict(
                key="a",
                image_id="i",
                text=r"\boxed{東京}",
                status="generated",
                capped=False,
            )
        ],
        Cache(),
    )[0]
    assert row["status"] == "unscored"
    assert row["metrics"]["content"] is None
