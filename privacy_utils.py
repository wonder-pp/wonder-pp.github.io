import re
from typing import Dict


def anonymize_label(item: Dict) -> str:
    student_id = str(item.get("学生编号", "")).strip().upper()
    if student_id.startswith("S") and student_id[1:].isdigit():
        return f"案例{student_id[1:]}"
    return "匿名案例"


def clean_experience_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""

    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"Q\d+\s*[:：][^\n]*A\s*[:：]\s*", "", value)
    value = re.sub(r"Q\s*[:：][^\n]*A\s*[:：]\s*", "", value)
    value = re.sub(r"Q\d+\s*[:：][^\n]*", "", value)
    value = re.sub(r"Q\s*[:：][^\n]*", "", value)
    value = re.sub(r"问\s*[:：][^\n]*答\s*[:：]\s*", "", value)
    value = re.sub(r"问\s*[:：][^\n]*", "", value)
    value = value.replace("A:", "").replace("A：", "").replace("答：", "").replace("答:", "")
    value = re.sub(r"\n{2,}", "\n", value)
    value = re.sub(r"\s{2,}", " ", value)
    value = value.strip(" \n\t-：:")

    fragments = [fragment.strip(" -：:") for fragment in re.split(r"[\n]", value) if fragment.strip()]
    cleaned = " ".join(fragments)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned
