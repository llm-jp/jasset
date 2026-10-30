"""OCR, QA and judge prompts used in the paper."""

OCR_PROMPT = "画像内のすべての文字を読み、説明を加えず、行ごとに改行して出力する"
VQA_SUFFIX = (
    "\n画像を参照して回答してください。推論過程は出力しても構いませんが、"
    "最終回答は必ず \\boxed{...} で囲み、ボックス内には最終回答のみを1つだけ記載してください。"
)
GRADER_TEMPLATE = """\
あなたは回答の正誤を判定する専門家です。
以下の[質問]に対する[正解]と、[モデルの回答]を比較し、正誤を判定してください。

[質問]: {question}
[正解]: {gold_answer}
[モデルの回答]: {prediction}

判定ルール:
- 表記の揺れは許容してください（例: 2羽 vs 二羽, 8,000 vs 8000）
- 数字の全角半角の違いは許容してください
- 正解の内容を含んでいれば、追加の文脈があっても正解としてください
- 正解と意味的に同じ回答であれば、言い回しが異なっていても正解としてください
- 単位が質問から自明な場合、省略されていても正解としてください

以下の形式のみで回答してください:
correct: yes
または
correct: no
"""


def task_prompt(task, row):
    if row.get("task_type") in ("ocr_fulltext", "qa_content"):
        return row["question"]
    return (
        OCR_PROMPT
        if task == "ocr"
        else row["question"]
        + (VQA_SUFFIX if "jawildtext" in row["dataset"].lower() else "")
    )
