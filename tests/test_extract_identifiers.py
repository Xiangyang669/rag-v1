"""正文标识符实体：`identifier` 节点 + `mentioned_in` 边。

背景：在此之前实体**只来自 `heading_path`**，正文一概不看。于是 FastGPT
中文文档里到处出现的 `MULTIPLE_DATA_TO_BASE64`、`config.json`、
`docker-compose.yml`、`projects/code-sandbox/src/env.ts` 一个都没进图，
用户拿这些名字提问时图谱路无物可匹配。

契约：
- 三类**形态判据**（全大写常量 / 带白名单扩展名的路径 / 成对反引号），
  **不做裸词抽取**——宁可漏抽，不可错抽。
- `identifier` 绝不进 `_STRUCTURAL_KINDS`，绝不成为 `belongs_to` 的一端。
- `aliases` 同时登记原样与小写；canonical 保持原样。
- 标识符必须进 posting list（`entities_of_chunk`）。
- 无标题的块：实体照抽，`mentioned_in` 不产（没有 section 可指）。
"""

from ragv1.ingest.extract import entities_of_chunk, extract
from ragv1.types import Chunk


def _chunk(cid, path, text):
    return Chunk(chunk_id=cid, doc_id="d", heading_path=tuple(path), text=text)


def _ids(ents):
    return {e.canonical for e in ents if e.kind == "identifier"}


# ── 判据一：全大写下划线常量 ──────────────────────────────────


def test_extracts_full_upper_snake_constant():
    ents, _ = extract([_chunk("c", ["配置"], "设 MULTIPLE_DATA_TO_BASE64 为 false")])
    assert "MULTIPLE_DATA_TO_BASE64" in _ids(ents)


def test_extracts_short_upper_with_underscore():
    ents, _ = extract([_chunk("c", ["配置"], "见 CHAT_MAX_QPM 的说明")])
    assert "CHAT_MAX_QPM" in _ids(ents)


def test_does_not_extract_common_short_uppercase():
    """AND / OK / FAQ 这类不该进图——它们会在几乎每个查询上命中。"""
    ents, _ = extract([_chunk("c", ["配置"], "A and B，OK 是 OK，FAQ 里说的")])
    assert _ids(ents) == set()


# ── 判据二：带白名单扩展名的文件名 / 路径 ──────────────────────


def test_extracts_filename_with_allowlisted_extension():
    ents, _ = extract([_chunk("c", ["部署"], "修改 docker-compose.yml 与 config.json")])
    assert {"docker-compose.yml", "config.json"} <= _ids(ents)


def test_extracts_path_with_extension():
    ents, _ = extract([_chunk("c", ["部署"], "见 projects/code-sandbox/src/env.ts")])
    assert "projects/code-sandbox/src/env.ts" in _ids(ents)


def test_does_not_extract_chinese_full_stop():
    ents, _ = extract([_chunk("c", ["概述"], "这是一句话。这是另一句。")])
    assert _ids(ents) == set()


# ── 判据三：成对反引号包裹的术语 ──────────────────────────────


def test_extracts_backticked_term():
    ents, _ = extract([_chunk("c", ["接口"], "传 `appId` 即可")])
    assert "appId" in _ids(ents)


def test_unpaired_backtick_is_not_extracted_and_does_not_raise():
    ents, _ = extract([_chunk("c", ["接口"], "落单的 ` 反引号")])
    assert _ids(ents) == set()


# ── Review Focus 3：identifier 不得污染层级 ────────────────────


def test_identifier_is_never_an_edge_endpoint():
    """⚠️ Review Focus 3：identifier 不得混进层级，否则污染后续命名。

    这条必须**真的能失败**——不能留一个恒真/恒空的断言：
    - `ids` 要恰好是抽出的那个标识符（抽空则说明断言在空转）；
    - `belongs_to` 端点要**非空且等于两个 section**（没有层级边则说明
      它在空转，而不是真的在检验污染）；
    - 最后断言它与标识符不相交。
    """
    ents, rels = extract([_chunk("c", ["配置", "子章节"], "设 MULTIPLE_DATA_TO_BASE64")])
    ids = _ids(ents)
    assert ids == {"MULTIPLE_DATA_TO_BASE64"}, "前提：本用例应当恰好抽出一个标识符"

    belongs = [r for r in rels if r.kind == "belongs_to"]
    endpoints = {r.src for r in belongs} | {r.dst for r in belongs}
    assert endpoints == {"配置", "子章节"}, "前提：层级边必须存在，否则断言空转"
    assert endpoints & ids == set()


# ── Review Focus 4：大小写两种写法命中同一节点 ────────────────


def test_both_cases_match_the_same_node():
    """⚠️ Review Focus 4：_normalize 会 lower()，两种写法都要能命中。"""
    ents, _ = extract([_chunk("c", ["配置"], "设 MULTIPLE_DATA_TO_BASE64 为 false")])
    ent = next(e for e in ents if e.canonical == "MULTIPLE_DATA_TO_BASE64")
    assert "multiple_data_to_base64" in ent.aliases


# ── posting list 与 mentioned_in ──────────────────────────────


def test_identifier_enters_posting_list():
    chunk = _chunk("c", ["配置"], "设 MULTIPLE_DATA_TO_BASE64 为 false")
    assert "MULTIPLE_DATA_TO_BASE64" in entities_of_chunk(chunk)


def test_mentioned_in_points_at_heading_leaf():
    _, rels = extract([_chunk("c", ["配置", "环境变量"], "设 MULTIPLE_DATA_TO_BASE64")])
    assert any(
        r.kind == "mentioned_in"
        and r.src == "MULTIPLE_DATA_TO_BASE64"
        and r.dst == "环境变量"
        for r in rels
    )


def test_chunk_without_heading_still_contributes_identifiers():
    """没有标题的块不该整块丢掉——它仍可能提到有用的标识符。"""
    ents, rels = extract([_chunk("c", (), "设 MULTIPLE_DATA_TO_BASE64 为 false")])
    assert "MULTIPLE_DATA_TO_BASE64" in _ids(ents)
    assert not [r for r in rels if r.kind == "mentioned_in"]  # 无 section 可指


# ── 确定性 ────────────────────────────────────────────────────


def test_extraction_is_deterministic_with_identifiers():
    chunks = [
        _chunk("c1", ["部署"], "改 config.json 与 docker-compose.yml，设 CHAT_MAX_QPM"),
        _chunk("c2", ["接口"], "传 `appId`，路径 projects/code-sandbox/src/env.ts"),
    ]
    a = extract(chunks)
    b = extract(chunks)
    assert a[0] == b[0]
    assert a[1] == b[1]
