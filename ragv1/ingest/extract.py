"""结构抽取：从块的结构与正文里抽出实体与关系。**纯规则，不得调用 LLM。**

实体有两个来源：
1. `heading_path`——章节 / 函数 / 类，构成层级（`belongs_to`）。
2. 块**正文**里形态明确的**标识符**（全大写常量 / 带白名单扩展名的路径 /
   成对反引号术语）——`identifier` 类型节点，不构成层级。

第 2 条**不是**「正文里出现某个词就是实体」：它只认三种高精度形态，
其余一律不抽。宁可漏抽，不可错抽——`AND`/`OK`/`FAQ` 这类常见大写词如果
进了图，会在几乎每个查询上命中，把图谱路变成噪声源。

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
# ⚠️ "identifier" **绝不**进 _STRUCTURAL_KINDS：那会让后续标题拿标识符做
# 命名限定，整条层级被污染。见 test_extract_hierarchy / test_extract_identifiers。

# ── 正文标识符的三类形态（模块级常量，可加不可减式扩充）─────────────
# 白名单：只有这些扩展名的「文件名 / 路径」才被当成标识符。
FILE_EXTENSIONS = (
    "json",
    "yml",
    "yaml",
    "toml",
    "ini",
    "conf",
    "env",
    "md",
    "txt",
    "sh",
    "ps1",
    "py",
    "js",
    "ts",
    "tsx",
    "jsx",
    "go",
    "java",
    "rs",
    "sql",
    "log",
)
_EXT_SET = frozenset(FILE_EXTENSIONS)

# 形态一 · 全大写下划线常量：MULTIPLE_DATA_TO_BASE64 / CHAT_MAX_QPM。
# 起止都要求不是词字符，避免从混合大小写的词里切出一段（URLPath 不抽）。
# 长度 ≥4 只是下限；真正的门在 `_constant_hits`——「含下划线 或 长度 ≥5」，
# 两样都占不到的（AND / OK / FAQ / GET / POST / HTTP）一律不抽。
_CONST_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9_]{3,}(?![A-Za-z0-9_])")

# 形态二 · 文件名 / 路径。用显式 ASCII 段，**不用 `\w`**——Python 的 `\w`
# 含 CJK，会把「见env.ts」的「见」也吞进 token。段结构也顺手挡掉以 `/`
# 或 `.` 起头（URL 的 `//example.com/x.ts` 只抽 `example.com/x.ts`）。
_PATH_RE = re.compile(r"[A-Za-z0-9_-]+(?:[./][A-Za-z0-9_-]+)*\.[A-Za-z0-9]+")

# 形态三 · 成对反引号内的术语。落单的反引号匹配不上，也不抛错。
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")


@dataclass(frozen=True)
class Entity:
    canonical: str
    kind: str  # section | function | class | parameter | identifier
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Relation:
    src: str
    dst: str
    kind: str  # has_parameter | belongs_to | mentioned_in


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


def _constant_hits(text: str) -> list[str]:
    """全大写常量命中的原文片段（保序、含重复），逐条做「含下划线或够长」的门。"""
    out: list[str] = []
    for m in _CONST_RE.finditer(text):
        token = m.group(0)
        if "_" in token or len(token) >= 5:
            out.append(token)
    return out


def _path_hits(text: str) -> list[str]:
    """带白名单扩展名的文件名 / 路径（保序、含重复）。"""
    out: list[str] = []
    for m in _PATH_RE.finditer(text):
        token = m.group(0)
        # 扩展名 = 最后一个 `.` 之后的部分；大小写不敏感地查白名单。
        ext = token.rsplit(".", 1)[-1]
        if ext.lower() in _EXT_SET:
            out.append(token)
    return out


def _backtick_hits(text: str) -> list[str]:
    """成对反引号内的非空术语（保序、含重复）。"""
    return [term.strip() for term in _BACKTICK_RE.findall(text) if term.strip()]


def _text_identifiers(text: str) -> list[str]:
    """从正文抽标识符原文，按出现顺序去重（三类形态之间也去重）。

    三类合并后去重：同一个词以多种形态出现（如 `` `config.json` `` 同时
    命中反引号与路径）只登记一次。
    """
    seen: dict[str, None] = {}
    for token in (*_constant_hits(text), *_path_hits(text), *_backtick_hits(text)):
        seen.setdefault(token, None)
    return list(seen)


def _identifier_entity(token: str) -> Entity:
    """标识符实体：canonical 保持原样（大小写是它的身份），别名登记小写。

    `_normalize` 内部会 `.lower()`，同时登记原样与小写后，检索器一行都不用
    改，`MULTIPLE_DATA_TO_BASE64` 与 `multiple_data_to_base64` 命中同一节点。
    """
    lower = token.lower()
    aliases = (token,) if lower == token else (token, lower)
    return Entity(canonical=token, kind="identifier", aliases=aliases)


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
    """把一块转成实体与关系：heading path 的层级 + 正文里的标识符。

    相邻两层 heading 之间产出一条 `belongs_to`（子 → 父）。src/dst 都用
    **该标题实体的 canonical**——与下面 `parent` 变量同一套取值——因此签名
    标题（`get(timeout=None)` → `Config.get`）会挂到它的限定名上，而不是
    字面标题，边与 section 树保持同构。顶层（`parent is None`）不产边。

    正文标识符另走一轮：每个标识符产出一个 `identifier` 实体；有标题时再产
    一条 `mentioned_in`（src=标识符，dst=heading path 的**叶子**）。**标识符
    绝不参与 `parent` 链**——它不在 `_STRUCTURAL_KINDS` 里，也不会被当成父标题。
    """
    entities: list[Entity] = []
    relations: list[Relation] = []
    parent: str | None = None
    for head in chunk.heading_path:
        ents, rels = _heading_entities(head, parent)
        entities.extend(ents)
        relations.extend(rels)
        for e in ents:
            if e.kind in _STRUCTURAL_KINDS:
                if parent is not None:
                    relations.append(Relation(src=e.canonical, dst=parent, kind="belongs_to"))
                parent = e.canonical

    # 正文标识符——走**独立**的一轮，绝不碰上面的 parent 链。无标题的块
    # 也能贡献标识符；只是没有 section 可指，不产 mentioned_in。
    leaf = chunk.heading_path[-1] if chunk.heading_path else None
    for token in _text_identifiers(chunk.text):
        entities.append(_identifier_entity(token))
        if leaf is not None:
            relations.append(Relation(src=token, dst=leaf, kind="mentioned_in"))
    return entities, relations


def entities_of_chunk(chunk: Chunk) -> list[str]:
    """该块上的全部实体 canonical（heading 层级 + 正文标识符）—— posting list 的来源。"""
    entities, _ = _chunk_graph(chunk)
    return sorted({e.canonical for e in entities})


def extract(chunks: list[Chunk]) -> tuple[list[Entity], list[Relation]]:
    """从一批块抽出全部实体与关系。

    无标题的块不再被整块跳过——它仍可能从正文里提到有用的标识符（只是
    没有 section 可指，不产 `mentioned_in`）。实体与关系都做了排序与去重，
    同一输入两次运行逐字节一致。
    """
    merged: dict[str, Entity] = {}
    seen_relations: set[tuple[str, str, str]] = set()

    for chunk in chunks:
        entities, relations = _chunk_graph(chunk)
        for e in entities:
            existing = merged.get(e.canonical)
            aliases = (
                e.aliases
                if existing is None
                else tuple(sorted(set(existing.aliases) | set(e.aliases)))
            )
            # 同形标识符与结构实体撞名时，结构类型优先：identifier 不得
            # 覆盖（也不得被误当成）章节/函数节点。
            kind = e.kind
            if (
                existing is not None
                and e.kind == "identifier"
                and existing.kind != "identifier"
            ):
                kind = existing.kind
            merged[e.canonical] = Entity(e.canonical, kind, aliases)
        for r in relations:
            seen_relations.add((r.src, r.dst, r.kind))

    ordered_entities = sorted(merged.values(), key=lambda e: (e.kind, e.canonical))
    ordered_relations = [Relation(s, d, k) for s, d, k in sorted(seen_relations)]
    return ordered_entities, ordered_relations
