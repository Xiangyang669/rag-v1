"""结构抽取：从块的结构里抽出实体与关系。**纯规则，不得调用 LLM。**

实体来自 `heading_path`，**不来自块正文**——正文里出现某个词不代表文档在
定义它。这也是「无标题的文档抽不出实体」自然成立的原因，而不是一个特例。

确定性：同一输入两次运行必须给出同样的实体与关系（都做了排序）。
"""

import re
from dataclasses import dataclass

from ragv1.types import Chunk

# 形如 name(params) 的签名标题
_SIG_RE = re.compile(r"^(?P<name>[^()]+)\((?P<params>.*)\)\s*$")
CLASS_PREFIX = "class "

# 会挂到 chunk 上的实体类型（parameter 是函数的附属物，不作为独立的标题层级）
_STRUCTURAL_KINDS = ("section", "function", "class")


@dataclass(frozen=True)
class Entity:
    canonical: str
    kind: str  # section | function | class | parameter
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Relation:
    src: str
    dst: str
    kind: str  # has_parameter


def _bare(name: str) -> str:
    """裸名：requests.get → get"""
    return name.rsplit(".", 1)[-1]


def _param_name(raw: str) -> str:
    """从 "timeout=None" / "*args" / "**kwargs" 里取参数名。"""
    return raw.split("=", 1)[0].strip().lstrip("*").strip()


def _heading_entities(head: str, parent: str | None) -> tuple[list[Entity], list[Relation]]:
    """把一个标题转成实体（以及它的参数实体与 has_parameter 关系）。"""
    head = head.strip()
    m = _SIG_RE.match(head)

    if not m:
        return [Entity(canonical=head, kind="section")], []

    raw_name = m.group("name").strip()
    is_class = raw_name.lower().startswith(CLASS_PREFIX)
    if is_class:
        raw_name = raw_name[len(CLASS_PREFIX) :].strip()

    # 已经是全限定名（含点）就原样用；否则用父标题限定，避免同名不同模块混淆
    canonical = raw_name if "." in raw_name else (f"{parent}.{raw_name}" if parent else raw_name)

    entities = [
        Entity(
            canonical=canonical,
            kind="class" if is_class else "function",
            aliases=(_bare(raw_name),),
        )
    ]
    relations: list[Relation] = []
    for piece in m.group("params").split(","):
        pname = _param_name(piece)
        if not pname:
            continue
        entities.append(Entity(canonical=pname, kind="parameter"))
        relations.append(Relation(src=canonical, dst=pname, kind="has_parameter"))

    return entities, relations


def _chunk_graph(chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
    """把一块的整条 heading path 转成实体与关系。"""
    entities: list[Entity] = []
    relations: list[Relation] = []
    parent: str | None = None
    for head in chunk.heading_path:
        ents, rels = _heading_entities(head, parent)
        entities.extend(ents)
        relations.extend(rels)
        for e in ents:
            if e.kind in _STRUCTURAL_KINDS:
                parent = e.canonical
    return entities, relations


def entities_of_chunk(chunk: Chunk) -> list[str]:
    """该块 heading path 上的全部实体 canonical —— posting list 的来源。"""
    entities, _ = _chunk_graph(chunk)
    return sorted({e.canonical for e in entities})


def extract(chunks: list[Chunk]) -> tuple[list[Entity], list[Relation]]:
    """从一批块抽出全部实体与关系。无标题的块不贡献任何实体。"""
    merged: dict[str, Entity] = {}
    seen_relations: set[tuple[str, str, str]] = set()

    for chunk in chunks:
        if not chunk.heading_path:
            continue
        entities, relations = _chunk_graph(chunk)
        for e in entities:
            existing = merged.get(e.canonical)
            aliases = (
                e.aliases
                if existing is None
                else tuple(sorted(set(existing.aliases) | set(e.aliases)))
            )
            merged[e.canonical] = Entity(e.canonical, e.kind, aliases)
        for r in relations:
            seen_relations.add((r.src, r.dst, r.kind))

    ordered_entities = sorted(merged.values(), key=lambda e: (e.kind, e.canonical))
    ordered_relations = [Relation(s, d, k) for s, d, k in sorted(seen_relations)]
    return ordered_entities, ordered_relations
