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
# 函数名/参数名必须是标识符。这条是**用真语料跑出来的**：
# 只看括号会把「Default Mode (Beta)」「Browser (Chrome / Firefox)」这类
# 普通标题误判成函数签名——合成测试里的 requests.get(...) 无歧义，看不出这问题。
_IDENT_RE = re.compile(r"[\w.]+")
_PARAM_RE = re.compile(r"\w+")
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


def _split_signature(head: str) -> tuple[str, list[str], bool] | None:
    """把标题解析成 (名字, 参数列表, 是否类)。**不像函数签名就返回 None。**

    两道门都必须过：
      1. 名字是标识符（真实函数名没有空格）
      2. 每个参数是标识符

    少任何一道，「Default Mode (Beta)」「Browser (Chrome / Firefox)」
    这类普通标题都会变成假函数。
    """
    m = _SIG_RE.match(head)
    if not m:
        return None

    name = m.group("name").strip()
    is_class = name.lower().startswith(CLASS_PREFIX)
    if is_class:
        name = name[len(CLASS_PREFIX) :].strip()

    if not _IDENT_RE.fullmatch(name):
        return None

    params: list[str] = []
    for piece in m.group("params").split(","):
        pname = _param_name(piece)
        if not pname:
            continue
        if not _PARAM_RE.fullmatch(pname):
            return None
        params.append(pname)

    return name, params, is_class


def _heading_entities(head: str, parent: str | None) -> tuple[list[Entity], list[Relation]]:
    """把一个标题转成实体（以及它的参数实体与 has_parameter 关系）。"""
    head = head.strip()

    signature = _split_signature(head)
    if signature is None:
        return [Entity(canonical=head, kind="section")], []

    raw_name, params, is_class = signature
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
    for pname in params:
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
