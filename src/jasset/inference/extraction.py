"""Extract final answers from model outputs.

These transformations preserve raw text in inference outputs.
"""


def final_answer(raw, prompt):
    for token in ("<|im_end|>", "<|endoftext|>"):
        if raw.rstrip().endswith(token):
            raw = raw.rstrip()[: -len(token)]
    # The official Thinking template can prefill <think>, absent from generated text.
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[1].strip(), "closed_thinking"
    if "<think>" in raw or prompt.rstrip().endswith("<think>"):
        return None, "unfinished_thinking"
    return raw.strip(), "plain_final"


def ocr_answer(final):
    import re

    if final is None:
        return None, "unfinished_thinking"
    found = re.findall(r"<answer>(.*?)</answer>", final, re.S)
    if found:
        return found[-1].strip(), "tagged"
    if "<answer>" in final:
        return final.rsplit("<answer>", 1)[1].strip(), "unclosed_answer"
    return final.strip(), "plain_final"
