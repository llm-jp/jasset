TASKS = {
    "full-ocr": "jawildtext_ocr",
    "crop": "crop",
    "cc-ocr": "cc_ocr_ja",
    "mtvqa": "mtvqa_qa",
}


def index(rows):
    indexed = {r["key"]: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate gold or prediction key")
    return indexed


def primary_gold(gold, cohort):
    refs = {r["key"]: r for r in gold}
    if len(refs) != len(gold):
        raise ValueError("Duplicate gold key")
    primary = {r["id"]: r for r in cohort if r["primary"]}
    if not primary.keys() <= refs.keys():
        raise ValueError("Incomplete gold coverage for primary cohort")
    for key, row in primary.items():
        if refs[key]["reference_unit"] != "region_strings":
            raise ValueError("Region ground truth required")
        if refs[key]["reference"] != [r["text"] for r in row["regions"]]:
            raise ValueError("Region references differ from frozen cohort")
    return {key: refs[key] for key in sorted(primary)}
