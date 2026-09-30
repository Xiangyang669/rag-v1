"""从语料结构自动生成评估集。**零人工标注、零 LLM。**

设计目标不是"覆盖文档"，而是**让每一路都有自己独占得分的问题**——
否则三路都答对，融合的增益就无从谈起：

| 类 | 生成方式 | 答案块 | 该赢的一路 |
|---|---|---|---|
| A 词面型 | 直接用标题文本转问句 | 该节的块 | 全文 |
| B 语义型 | 用 synonyms 规则替换标题词 | 该节的块 | 向量 |
| C 聚合型 | 取跨越 ≥3 块的实体 | 该实体邻居块的并集 | **图谱** |
| D 归属型 | 取某个 section | 该 section 子树下全部块 | **图谱** |

C/D 的共同点是**答案是一组块而不是一个块**——向量路和全文路本质上都是
"块打分排序"，天然只返回孤立的块；跨块聚合只有图能直接走。

全部输出都做排序，同一输入两次运行必须完全一致。
"""

from dataclasses import dataclass

from ragv1.evaluation.synonyms import SYNONYMS
from ragv1.store.graph_store import GraphStore
from ragv1.types import Chunk

CATEGORY_A = "A"
CATEGORY_B = "B"
CATEGORY_C = "C"
CATEGORY_D = "D"

# C 类要求实体跨越的块数下限。
# 只跨 2 块的问题，向量路碰运气也能凑齐（天花板效应），测不出图谱的价值。
MIN_CHUNKS_FOR_C = 3


@dataclass(frozen=True)
class QAItem:
    question: str
    answer_chunk_ids: tuple[str, ...]
    category: str  # "A" | "B" | "C" | "D"


def _rewrite_with_synonym(text: str) -> tuple[str, bool]:
    """用同义词表改写。取第一个命中的键，保证可复现。"""
    for key, values in SYNONYMS.items():
        if key in text:
            return text.replace(key, values[0]), True
    return text, False


def _gen_a(chunks: list[Chunk], limit: int) -> list[QAItem]:
    out: list[QAItem] = []
    for chunk in chunks:
        if not chunk.heading_path:
            continue
        out.append(
            QAItem(
                question=f"{chunk.heading_path[-1]} 是什么？",
                answer_chunk_ids=(chunk.chunk_id,),
                category=CATEGORY_A,
            )
        )
        if len(out) >= limit:
            break
    return out


def _gen_b(chunks: list[Chunk], limit: int) -> list[QAItem]:
    out: list[QAItem] = []
    for chunk in chunks:
        if not chunk.heading_path:
            continue
        rewritten, changed = _rewrite_with_synonym(chunk.heading_path[-1])
        if not changed:
            continue
        out.append(
            QAItem(
                question=f"我想了解 {rewritten} 相关的内容",
                answer_chunk_ids=(chunk.chunk_id,),
                category=CATEGORY_B,
            )
        )
        if len(out) >= limit:
            break
    return out


def _canonical_entities(graph: GraphStore) -> list[str]:
    """枚举图中的规范实体名。

    canonical 在别名表里映射到自身，别名则映射到 canonical——
    所以 `k == v` 的那些键就是规范实体。
    """
    mapping = graph.aliases()
    return sorted(k for k, v in mapping.items() if k == v)


def _gen_c(graph: GraphStore, limit: int) -> list[QAItem]:
    out: list[QAItem] = []
    for entity in _canonical_entities(graph):
        own = graph.chunks_of(entity)
        if len(own) < MIN_CHUNKS_FOR_C:
            continue
        answer = set(own)
        for neighbour in graph.neighbors(entity):
            answer.update(graph.chunks_of(neighbour))
        out.append(
            QAItem(
                question=f"{entity} 出现在哪些地方？",
                answer_chunk_ids=tuple(sorted(answer)),
                category=CATEGORY_C,
            )
        )
        if len(out) >= limit:
            break
    return out


def _gen_d(chunks: list[Chunk], limit: int) -> list[QAItem]:
    # 祖先路径即 section 链（heading_path 的最后一个元素才是本块自己的标题）
    prefixes = {c.heading_path[:i] for c in chunks for i in range(1, len(c.heading_path))}

    out: list[QAItem] = []
    for prefix in sorted(prefixes):
        answer = tuple(
            sorted(c.chunk_id for c in chunks if c.heading_path[: len(prefix)] == prefix)
        )
        out.append(
            QAItem(
                question=f"{prefix[-1]} 下都讲了什么？",
                answer_chunk_ids=answer,
                category=CATEGORY_D,
            )
        )
        if len(out) >= limit:
            break
    return out


def generate_qas(chunks: list[Chunk], graph: GraphStore, per_category: int) -> list[QAItem]:
    """生成四类问题，每类最多 per_category 条。结果已排序，可复现。"""
    ordered = sorted(chunks, key=lambda c: c.chunk_id)

    items = [
        *_gen_a(ordered, per_category),
        *_gen_b(ordered, per_category),
        *_gen_c(graph, per_category),
        *_gen_d(ordered, per_category),
    ]
    return sorted(items, key=lambda q: (q.category, q.question))
