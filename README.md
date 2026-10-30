# JaSSeT

日本語合成シーンテキストを用いた、OCRと画像質問応答のマルチタスク学習。

[データセット](https://huggingface.co/datasets/llm-jp/jasset) · [論文サイト](https://llm-jp.github.io/jasset/)

## 準備

Python 3.11–3.12を使用する。データセットは利用申請の承認後に取得できる。

```bash
uv sync --locked --extra dev
uv run hf auth login
uv run python -m jasset.prepare --output data/jasset
```

## 学習

Linux/CUDA環境で実行する。

```bash
uv sync --locked --extra train
uv run --extra train torchrun --standalone --nproc_per_node=4 \
  --module jasset.train \
  --config configs/train/joint.yaml --data data/jasset --output runs/joint
```

`configs/train/` にOCRのみ、QAのみ、共同学習、根拠報酬なしの4条件を用意している。学習結果は `runs/joint/export/` に保存する。

推論は `uv run python -m jasset.predict --help`、評価は `uv run python -m jasset.evaluate --help` を参照。テストは `uv run pytest -q` で実行する。

第三者素材の出典・ライセンスは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照。研究コードの利用ライセンスは未設定。
