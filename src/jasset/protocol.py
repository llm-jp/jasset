"""Generation settings for evaluation input preparation."""

CROP_PROMPT = "画像内の文字をそのまま書き起こしてください。説明は不要です。"
GENERATION = dict(
    max_pixels=1605632,
    min_pixels=65536,
    max_tokens=4096,
    max_model_len=32768,
    temperature=0,
    seed=0,
    enable_thinking=False,
)
CROP_GENERATION = dict(GENERATION, max_tokens=512, max_model_len=4096)
