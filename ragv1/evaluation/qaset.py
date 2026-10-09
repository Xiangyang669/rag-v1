"""人工评估集的加载器。

评估集是 JSONL：每行一个评测项，形如

    {"question": "...", "answer_chunk_ids": ["a", "b"], "category": "fact",
     "history": [{"role": "user", "text": "..."}, ...]}

这是**人工撰写**的评估集——后面关于答案质量的每一个结论都以它为准，
因此格式校验宁严勿松：缺任一必需字段都当作格式错误抛出，
而不是用默认值糊过去，否则一个笔误就会静默地扭曲分数。
"""

import json
from dataclasses import dataclass
from pathlib import Path

from ragv1.types import Turn


class QASetError(Exception):
    """评估集格式错误。消息带行号，便于定位手写数据里的笔误。"""


@dataclass(frozen=True)
class EvalItem:
    question: str
    answer_chunk_ids: tuple[str, ...]
    category: str
    history: tuple[Turn, ...] = ()


# 必需字段。answer_chunk_ids 允许为空（unanswerable 类问题没有答案块）。
_REQUIRED = ("question", "answer_chunk_ids", "category")


def _parse_row(row: dict, lineno: int) -> EvalItem:
    for field in _REQUIRED:
        if field not in row:
            raise QASetError(f"第 {lineno} 行：缺少 {field}")

    history = tuple(
        Turn(role=t["role"], text=t["text"]) for t in row.get("history", ())
    )
    return EvalItem(
        question=row["question"],
        answer_chunk_ids=tuple(row["answer_chunk_ids"]),
        category=row["category"],
        history=history,
    )


def load_qaset(path: str | Path) -> tuple[EvalItem, ...]:
    """逐行解析 JSONL。空白行跳过；任一行的格式错误以 QASetError 抛出。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    items: list[EvalItem] = []
    for lineno, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as err:
            raise QASetError(f"第 {lineno} 行：{err}") from err
        items.append(_parse_row(row, lineno))
    return tuple(items)
