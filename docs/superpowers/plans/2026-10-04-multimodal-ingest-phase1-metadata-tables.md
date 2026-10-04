# 多模态入库 · 阶段一：元数据骨架 + 表格解析 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让表格从正文中拆出、结构化、独立成 chunk 且绝不被字符硬切；同时给每个 chunk 装上端到端可读的来源元数据。

**Architecture:** 引入 `Element` 作为版面分析与分块之间的中间表示。`MarkdownLoader` 产出有序 Element 流，`elements.chunk_elements` 消费它并复用现有 `chunker` 的正文切分逻辑。同一个 `heading_path` 下连续的 text Element 会先合并回一个正文段，因此**纯文本文档的产出与改造前逐字节一致**。表格走下一条独立分支：整表成一 chunk，超限按「表头+若干行」切且每块重复表头。

**Tech Stack:** Python 3.12 · pytest · SQLite FTS5 · Chroma。**本阶段不引入任何新依赖。**

**Spec:** `docs/superpowers/specs/2026-10-04-multimodal-ingest-design.md`

## Global Constraints

- **本阶段零新依赖**：`requirements.txt` 不动
- **Python 3.12**，本机解释器命令是 `py`（`python` / `python3` / `pytest` 都不在 PATH）
- 测试命令一律为 `py -m pytest`（在 `rag-v1/` 目录下执行）
- **现有 107 个测试必须全程全绿**，基线：`107 passed in 5.61s`
- `Chunk` 新增字段**全部带默认值**——现有 `Chunk(...)` 构造点一个都不许改
- `make_chunk_id(doc_id, heading_path, index)` **签名不变**，`index` 仍是全局递增序号
- 所有代码注释、docstring、错误信息用**中文**
- 降级原因码**只能**取自 `ragv1/ingest/degrade.py` 的 `ALL_CODES`
- 任何解析失败**必须落降级码**，不允许静默丢弃
- 每个 chunk 的 `degrade` **有序去重**，保证同输入两次运行输出完全一致

## Review Focus

以下五类输入是 spec 未逐条点名、但最可能咬到使用者的。每条都在下方任务里配了测试。

1. **代码围栏里的表格行** —— 技术文档里代码块遍地；``` 内的 `|` 行若被当成表格，会把代码从正文里挖走
2. **分隔行与表头列数不一致的畸形表** —— 会被切歪或抛异常；期望是降级保留原文而非崩溃
3. **只有表头+分隔行、零数据行的空表** —— 期望原样成一块，而不是产出空块或抛 `IndexError`
4. **单元格内含转义竖线 `\|`** —— 朴素 `split("|")` 会多切出一列
5. **单行本身就超预算的超长表行** —— 期望该行独立成块并落 `table_row_too_long`，而不是被切断

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `ragv1/ingest/degrade.py` | 新建 | 降级原因码常量表 + 有序去重工具 |
| `ragv1/types.py` | 修改 | 新增 `Element`；`Chunk` 扩元数据字段 |
| `ragv1/ingest/table.py` | 新建 | Markdown 表识别 / 规范化 / 表头重复切分 / 降级 |
| `ragv1/ingest/parser.py` | 修改 | 把围栏判定与标题正则暴露为可复用函数（**不改现有行为**） |
| `ragv1/ingest/loader.py` | 新建 | `MarkdownLoader`：md → 有序 Element 流（本阶段只做 text + table） |
| `ragv1/ingest/elements.py` | 新建 | Element 流 → `list[Chunk]`（文本段合并 + 表格独立分支） |
| `ragv1/store/fts_store.py` | 修改 | `chunks` 表加元数据列 + `schema_version` 守卫 + `meta_of` |
| `ragv1/store/vector_store.py` | 修改 | `add()` 传 Chroma `metadatas` |
| `ragv1/ingest/build.py` | 修改 | 接入新链路（loader + elements） |
| `ragv1/api/main.py` | 修改 | `create_app(meta_lookup=...)` + `ResultItem` 带出来源 |
| `tests/fixtures/table_doc.md` | 新建 | 含小表 + 大表 + 正常正文的公开 fixture |

**阶段边界**：本阶段 `MarkdownLoader` **不识别图片**——`![alt](url)` 行继续留在正文里，与今天行为一致。图片元素识别与双通道留到阶段二，因此本阶段只有 23 篇含表文档的 `chunk_id` 会移位。

---

### Task 1: 降级原因码表

**Files:**
- Create: `ragv1/ingest/degrade.py`
- Test: `tests/test_degrade_codes.py`

**Interfaces:**
- Consumes: 无
- Produces: 14 个字符串常量 + `ALL_CODES: frozenset[str]` + `normalize(codes: Iterable[str]) -> tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_degrade_codes.py
from ragv1.ingest import degrade


def test_all_codes_covers_every_constant():
    """ALL_CODES 必须覆盖模块里定义的每一个码，否则测试断言形同虚设。"""
    defined = {
        v for k, v in vars(degrade).items()
        if not k.startswith("_") and isinstance(v, str)
    }
    assert defined == set(degrade.ALL_CODES)


def test_normalize_sorts_and_dedupes():
    """有序去重是「同输入两次运行输出一致」的依据。"""
    assert degrade.normalize(["vlm_failed", "ocr_empty", "vlm_failed"]) == (
        "ocr_empty", "vlm_failed",
    )


def test_normalize_empty():
    assert degrade.normalize([]) == ()


def test_all_codes_has_the_fourteen():
    assert len(degrade.ALL_CODES) == 14
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_degrade_codes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragv1.ingest.degrade'`

- [ ] **Step 3: Implement `ragv1/ingest/degrade.py`**

模块级常量（名字与值逐字照抄 spec §6.4），值为其小写蛇形名：

```python
TABLE_UNSTRUCTURED     = "table_unstructured"
TABLE_ROW_TOO_LONG     = "table_row_too_long"
IMAGE_REF_UNRESOLVED   = "image_ref_unresolved"
IMAGE_FETCH_FAILED     = "image_fetch_failed"
IMAGE_MISSING          = "image_missing"
OCR_FAILED             = "ocr_failed"
OCR_EMPTY              = "ocr_empty"
OCR_LOW_CONF           = "ocr_low_conf"
VLM_UNAVAILABLE        = "vlm_unavailable"
VLM_FAILED             = "vlm_failed"
VLM_BAD_JSON           = "vlm_bad_json"
PDF_PAGE_NO_TEXT_LAYER = "pdf_page_no_text_layer"
PDF_RENDER_FAILED      = "pdf_render_failed"
PDF_TABLE_BBOX_MISSING = "pdf_table_bbox_missing"
```

`ALL_CODES` 由这些常量显式列出（不要用 `vars()` 自省——测试要用它来校验，自省会让测试变成同义反复）。`normalize` 用已排序的 `sorted(set(...))`。

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_degrade_codes.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Run the whole suite**

Run: `py -m pytest -q`
Expected: `111 passed`

- [ ] **Step 6: Commit**

```bash
git add ragv1/ingest/degrade.py tests/test_degrade_codes.py
git commit -m "feat(ingest): 降级原因码常量表与有序去重工具"
```

---

### Task 2: `Element` 与 `Chunk` 元数据字段

**Files:**
- Modify: `ragv1/types.py`
- Test: `tests/test_elements.py`（本任务只放类型相关用例；分块用例在 Task 5 追加）

**Interfaces:**
- Consumes: 无
- Produces:
  - `Element(kind: str, order: int, text: str, heading_path: tuple[str, ...] = (), page: int | None = None, bbox: tuple[float,float,float,float] | None = None, image_ref: str | None = None, image_path: str | None = None, table_structured: bool = True, degrade: tuple[str, ...] = ())`
  - `Chunk` 追加字段：`kind="text"`, `page=None`, `order=0`, `part=0`, `bbox=None`, `image_ref=None`, `image_path=None`, `table_structured=True`, `degrade=()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_elements.py
from ragv1.types import Chunk, Element


def test_chunk_old_construction_still_works():
    """向后兼容是硬约束：现有构造点一个都不许改。"""
    c = Chunk(chunk_id="a", doc_id="d.md", heading_path=("H",), text="正文")
    assert c.kind == "text"
    assert c.page is None and c.order == 0 and c.part == 0
    assert c.bbox is None and c.image_ref is None and c.image_path is None
    assert c.table_structured is True
    assert c.degrade == ()


def test_element_defaults():
    e = Element(kind="text", order=0, text="正文")
    assert e.heading_path == () and e.page is None and e.bbox is None
    assert e.table_structured is True and e.degrade == ()


def test_element_and_chunk_are_frozen():
    import dataclasses
    assert dataclasses.fields(Chunk)  # 存在即通过；frozen 由下面的赋值验证
    e = Element(kind="text", order=0, text="x")
    try:
        e.text = "y"
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("Element 必须是 frozen 的")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_elements.py -v`
Expected: FAIL — `ImportError: cannot import name 'Element'`

- [ ] **Step 3: Implement in `ragv1/types.py`**

新增 `Element` dataclass（逐字照抄 spec §4.1，含中文 docstring）。`Chunk` 追加九个字段（照抄 spec §4.2），**全部带默认值**，并在字段上加分组注释标明「来源元数据」。保持 `@dataclass(frozen=True)`。

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_elements.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Run the whole suite to prove nothing broke**

Run: `py -m pytest -q`
Expected: `114 passed` — **这一步是本任务的核心**；若有现有测试因字段顺序变化而失败，说明有位置参数构造点，必须修掉而不是改默认值

- [ ] **Step 6: Commit**

```bash
git add ragv1/types.py tests/test_elements.py
git commit -m "feat(types): 新增 Element 中间表示，Chunk 扩展来源元数据字段"
```

---

### Task 3: 表格识别、规范化与表头重复切分

**Files:**
- Create: `ragv1/ingest/table.py`
- Test: `tests/test_table.py`

**Interfaces:**
- Consumes: `ragv1.ingest.degrade`
- Produces:
  - `is_table_line(line: str) -> bool`
  - `is_separator_line(line: str) -> bool`
  - `split_row(line: str) -> list[str]`
  - `normalize_table(raw_lines: list[str]) -> str | None`
  - `split_table(md: str, budget: int) -> tuple[list[str], tuple[str, ...]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_table.py
import pytest

from ragv1.ingest import degrade
from ragv1.ingest.table import normalize_table, split_row, split_table


def test_is_separator_and_table_lines():
    from ragv1.ingest.table import is_separator_line, is_table_line
    assert is_table_line("| a | b |")
    assert is_table_line("  | a | b |")          # 允许前导空白（语料实测存在）
    assert not is_table_line("文本 | 竖线")
    assert is_separator_line("| --- | :-- |")
    assert is_separator_line("|---|")
    assert not is_separator_line("| a | b |")


def test_split_row_handles_escaped_pipe():
    """转义竖线不能多切出一列 —— Review Focus #4"""
    assert split_row(r"| a \| b | c |") == ["a | b", "c"]


def test_normalize_strips_cells_and_keeps_shape():
    assert normalize_table(["| a |b|", "|---|---|", "| 1 | 2 |"]) == (
        "| a | b |\n| --- | --- |\n| 1 | 2 |"
    )


def test_normalize_rejects_malformed_column_count():
    """分隔行列数与表头不一致 —— Review Focus #2 → 返回 None 让调用方降级"""
    assert normalize_table(["| a | b |", "|---|", "| 1 | 2 |"]) is None


def test_normalize_rejects_when_second_line_is_not_separator():
    assert normalize_table(["| a | b |", "| 1 | 2 |"]) is None


def test_split_table_fits_in_one_block():
    md = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    parts, codes = split_table(md, budget=1000)
    assert parts == [md]
    assert codes == ()


def test_split_table_repeats_header_on_every_part():
    rows = [f"| {i} | {'x' * 20} |" for i in range(20)]
    md = "\n".join(["| a | b |", "| --- | --- |", *rows])
    parts, codes = split_table(md, budget=200)
    assert len(parts) > 1
    for p in parts:
        assert p.splitlines()[0] == "| a | b |"      # 每块都带表头
        assert p.splitlines()[1] == "| --- | --- |"
    assert codes == ()
    assert "\n".join(parts).count("| x") == 20 * 1   # 一行都没丢


def test_split_table_row_longer_than_budget_is_not_cut():
    """Review Focus #5：宁可超限也不切断行"""
    long_row = "| " + "y" * 500 + " |"
    md = "\n".join(["| a |", "| --- |", long_row])
    parts, codes = split_table(md, budget=50)
    assert any(long_row in p for p in parts)          # 整行完整出现在某一块
    assert degrade.TABLE_ROW_TOO_LONG in codes


def test_split_table_header_only_table_is_one_block():
    """Review Focus #3：零数据行的空表不能抛异常，也不能产出空块"""
    md = "| a | b |\n| --- | --- |"
    parts, codes = split_table(md, budget=1000)
    assert parts == [md]
    assert codes == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_table.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragv1.ingest.table'`

- [ ] **Step 3: Implement `ragv1/ingest/table.py`**

要点（签名与上面的测试已确定行为，实现细节自行决定）：

- `split_row`：按**未转义**的 `|` 切分，再还原 `\|` → `|`，去掉首尾空串与每格两侧空白
- `normalize_table`：`len(raw_lines) < 2` 或第 2 行不是分隔行 → `None`；**表头列数与分隔行列数必须相等**，否则 `None`；每格 `strip()` 后用 `" | "` 与 `"| "` / `" |"` 重新拼成规范形态；分隔行统一重写成 `| --- |` 形式（列数对齐）
- `split_table`：第 0 行是表头、第 1 行是分隔行、其余为数据行；`header = f"{row0}\n{row1}"`；贪心装行；**单行自身超预算**时先把当前块（若已含数据行）落盘，再把该行单独成块并追加 `TABLE_ROW_TOO_LONG`；零数据行时整表原样返回一块
- 返回前对降级码调用 `degrade.normalize`

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_table.py -v`
Expected: PASS（9 passed）

- [ ] **Step 5: Run the whole suite**

Run: `py -m pytest -q`
Expected: `123 passed`

- [ ] **Step 6: Commit**

```bash
git add ragv1/ingest/table.py tests/test_table.py
git commit -m "feat(ingest): 表格识别、规范化与表头重复切分"
```

---

### Task 4: `MarkdownLoader` —— md → 有序 Element 流

**Files:**
- Modify: `ragv1/ingest/parser.py`（仅暴露可复用函数，不改现有行为）
- Create: `ragv1/ingest/loader.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: `ragv1.types.Element`、`ragv1.ingest.table`、`ragv1.ingest.degrade`
- Produces:
  - `ragv1.ingest.parser.is_fence_line(line: str) -> bool`（新增）
  - `ragv1.ingest.parser.strip_frontmatter(text: str) -> str`（由现有 `_strip_frontmatter` 更名导出，保留 `_strip_frontmatter` 别名以免破坏调用点）
  - `ragv1.ingest.loader.load_document(path: Path, doc_id: str) -> list[Element]`
  - `ragv1.ingest.loader.markdown_elements(text: str, doc_id: str) -> list[Element]`

**注意：本阶段不识别图片。** `![alt](url)` 行按普通正文行处理，与今天一致。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loader.py
from pathlib import Path

from ragv1.ingest import degrade
from ragv1.ingest.loader import load_document, markdown_elements


def test_headings_set_heading_path_and_text_is_split_at_table():
    md = "# H\n\n前段。\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n后段。\n"
    els = markdown_elements(md, doc_id="d.md")
    assert [e.kind for e in els] == ["text", "table", "text"]
    assert els[0].heading_path == ("H",)
    assert els[0].text == "前段。"
    assert els[1].text.startswith("| a | b |")
    assert els[2].text == "后段。"
    assert [e.order for e in els] == [0, 1, 2]


def test_table_inside_code_fence_is_not_a_table():
    """Review Focus #1：代码块里的 | 行不是表格"""
    md = "# H\n\n```bash\n| a | b |\n| --- | --- |\n| 1 | 2 |\n```\n"
    els = markdown_elements(md, doc_id="d.md")
    assert [e.kind for e in els] == ["text"]
    assert "| --- | --- |" in els[0].text


def test_image_line_stays_in_text_this_phase():
    """阶段一边界：图片行不识别，继续留在正文里"""
    md = "# H\n\n![alt](http://x/y.png)\n"
    els = markdown_elements(md, doc_id="d.md")
    assert [e.kind for e in els] == ["text"]
    assert "![alt](http://x/y.png)" in els[0].text


def test_frontmatter_is_stripped():
    md = "---\ntitle: X\n---\n# H\n\n正文。\n"
    els = markdown_elements(md, doc_id="d.md")
    assert "title: X" not in els[0].text
    assert els[0].heading_path == ("H",)


def test_multiple_headings_produce_nested_paths():
    md = "# A\n\n一。\n\n## B\n\n二。\n\n# C\n\n三。\n"
    els = markdown_elements(md, doc_id="d.md")
    assert [e.heading_path for e in els] == [("A",), ("A", "B"), ("C",)]


def test_malformed_table_degrades_but_keeps_raw_text():
    """Review Focus #2：畸形表降级保留原文，不丢"""
    md = "# H\n\n| a | b |\n|---|\n| 1 | 2 |\n"
    els = markdown_elements(md, doc_id="d.md")
    assert [e.kind for e in els] == ["text"]         # 没被当成表格
    assert "| 1 | 2 |" in els[0].text


def test_load_document_dispatches_by_suffix(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("# H\n\n正文。\n", encoding="utf-8")
    assert len(load_document(p, doc_id="a.md")) == 1


def test_load_document_unknown_suffix_returns_empty(tmp_path: Path):
    p = tmp_path / "a.json"
    p.write_text("{}", encoding="utf-8")
    assert load_document(p, doc_id="a.json") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragv1.ingest.loader'`

- [ ] **Step 3: Expose the two helpers in `ragv1/ingest/parser.py`**

把现有的围栏判定抽成模块级 `is_fence_line(line)` 并在 `parse_document` 里改用它（行为等价：原先的 `line.lstrip().startswith("```") or line.lstrip().startswith("~~~")`）。把 `_strip_frontmatter` 更名导出为 `strip_frontmatter`，保留 `_strip_frontmatter = strip_frontmatter` 一行别名。

Run: `py -m pytest tests/test_parser.py -v`
Expected: PASS（4 passed）—— **先证明重构没破坏 parser 再往下写**

- [ ] **Step 4: Implement `ragv1/ingest/loader.py`**

`markdown_elements(text, doc_id)` 的实现要点：

- 复用 `parser.strip_frontmatter` 与 `parser.is_fence_line`，标题正则用 `parser._HEADING_RE`
- 维护标题栈（与 `chunker._iter_sections` 同一套：`while len(stack) >= level: stack.pop()` 再 `append`）
- 逐行扫描；**不在围栏内**时，遇到标题行 / 表格起点 / 文档结束就 flush 当前正文缓冲
- **表格起点判定**：当前行 `is_table_line` 且下一行 `is_separator_line` → 向后吃掉所有连续的 `is_table_line` 行，交给 `normalize_table`；返回 `None`（畸形表）则**把这批行原样并回正文缓冲**（这就是畸形表降级的实现方式）
- flush 时正文按 `strip()` 后非空才产出 `Element(kind="text", order=len(elements), ...)`；`order` 用产出序号
- table Element：`Element(kind="table", text=normalized, table_structured=True, degrade=())`
- `load_document(path, doc_id)`：`.md` / `.markdown` → 读文本走 `markdown_elements`；其他后缀返回 `[]`（**阶段二在此加 `.pdf` 分支**）

- [ ] **Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_loader.py -v`
Expected: PASS（8 passed）

- [ ] **Step 6: Run the whole suite**

Run: `py -m pytest -q`
Expected: `131 passed`

- [ ] **Step 7: Commit**

```bash
git add ragv1/ingest/parser.py ragv1/ingest/loader.py tests/test_loader.py
git commit -m "feat(ingest): MarkdownLoader 产出有序 Element 流（表格独立于正文）"
```

---

### Task 5: `chunk_elements` —— Element 流 → Chunk，含纯文本等价性回归

**Files:**
- Create: `ragv1/ingest/elements.py`
- Test: `tests/test_elements.py`（追加）

**Interfaces:**
- Consumes: `ragv1.types.{Chunk, Element}`、`ragv1.ingest.{chunker, table, degrade, loader}`
- Produces: `chunk_elements(elements: list[Element], doc_id: str, max_chars: int) -> list[Chunk]`

- [ ] **Step 1: Write the failing test**

```python
# 追加到 tests/test_elements.py
from ragv1.config import MAX_CHARS
from ragv1.ingest.chunker import chunk_document
from ragv1.ingest.elements import chunk_elements
from ragv1.ingest.loader import markdown_elements
from ragv1.ingest.parser import parse_document

_PURE_TEXT_DOCS = [
    # 无表格、无图片：覆盖多级标题、空节、代码围栏、超长节
    "# A\n\n前段。\n\n## B\n\n二段。\n\n# C\n\n三段。\n",
    "无标题文档，只有一段正文。\n",
    "# H\n\n```python\n# 代码里的井号不是标题\nx = 1\n```\n\n正文。\n",
    "# H\n\n" + "很长的段落。" * 300 + "\n",          # 触发 OVER_CAP 再切
    "# A\n\n一。\n\n## B\n\n",                          # 空节
]


def test_pure_text_output_is_byte_identical_to_old_pipeline():
    """硬约束：不含表格/图片的文档，新旧链路产出逐字节一致。

    这条是「不破坏现有纯文本文档解析路径」的可执行证明。
    """
    for md in _PURE_TEXT_DOCS:
        old = chunk_document(parse_document(md, doc_id="d.md"), max_chars=MAX_CHARS)
        new = chunk_elements(markdown_elements(md, doc_id="d.md"), "d.md", MAX_CHARS)
        assert [(c.chunk_id, c.text) for c in new] == [
            (c.chunk_id, c.text) for c in old
        ], f"纯文本文档产出不一致：{md[:40]!r}"


def test_table_gets_its_own_chunk():
    md = "# H\n\n前段。\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n后段。\n"
    chunks = chunk_elements(markdown_elements(md, doc_id="d.md"), "d.md", MAX_CHARS)
    kinds = [c.kind for c in chunks]
    assert kinds.count("table") == 1
    table_chunk = chunks[kinds.index("table")]
    assert table_chunk.table_structured is True
    assert "| 1 | 2 |" in table_chunk.text
    # 表格不在任何正文块里（"不和正文混在一起"）
    assert all("| 1 | 2 |" not in c.text for c in chunks if c.kind != "table")


def test_oversized_table_is_not_char_cut_and_every_part_has_header():
    """需求 2 的核心：大表按表头+若干行切，绝不被 _pieces_within 硬切"""
    rows = [f"| {i} | {'x' * 40} |" for i in range(30)]
    md = "# H\n\n" + "\n".join(["| a | b |", "| --- | --- |", *rows]) + "\n"
    chunks = chunk_elements(markdown_elements(md, doc_id="d.md"), "d.md", MAX_CHARS)
    table_chunks = [c for c in chunks if c.kind == "table"]
    assert len(table_chunks) > 1
    for c in table_chunks:
        assert c.text.splitlines()[1] == "| --- | --- |"   # 每块都带表头
    assert [c.part for c in table_chunks] == list(range(len(table_chunks)))
    # 30 行一行不少，且没有任何一行被切断
    assert "\n".join(c.text for c in table_chunks).count("| x") == 30


def test_chunk_ids_are_deterministic():
    """同输入两次运行必须给出同样的 chunk_id，否则跨重启对不回原文"""
    md = "# H\n\n前段。\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    a = chunk_elements(markdown_elements(md, doc_id="d.md"), "d.md", MAX_CHARS)
    b = chunk_elements(markdown_elements(md, doc_id="d.md"), "d.md", MAX_CHARS)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_metadata_is_carried_onto_chunks():
    md = "# H\n\n前段。\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    chunks = chunk_elements(markdown_elements(md, doc_id="d.md"), "d.md", MAX_CHARS)
    table_chunk = next(c for c in chunks if c.kind == "table")
    assert table_chunk.doc_id == "d.md"
    assert table_chunk.heading_path == ("H",)
    assert table_chunk.order == 1                      # 与 Element 的 order 对齐
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_elements.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragv1.ingest.elements'`

- [ ] **Step 3: Implement `ragv1/ingest/elements.py`**

算法（签名与测试已定行为）：

1. **合并连续 text Element**：同一 `heading_path` 下相邻的 `kind == "text"` 元素，用 `"\n\n"` 连接成一个正文段（保留第一个元素的 `order`、`page`、`bbox`）。表格/图片是断开点。**这一步是纯文本等价性的来源**——合并后的段与 `chunker._iter_sections` 产出的 body 完全相同
2. 逐段产出 Chunk：
   - **text 段** → 直接调 `chunker._group_text(heading_path, body, max_chars)`（复用现有实现，含 OVER_CAP 再切），每块 `kind="text"`
   - **table 段** → `budget = max_chars - len(chunker._head(heading_path))`；`table.split_table(text, budget)` 得 `(parts, codes)`；每块前缀 `chunker._head(heading_path)`，`part` 为序号，`table_structured=True`，`degrade=codes`。**不走 `_pieces_within`**
   - **image 段**（阶段二才会产生）→ 不切，整段一块
3. `chunk_id` 一律 `chunker.make_chunk_id(doc_id, heading_path, len(chunks))`——**全局递增 `index`，签名不变**
4. 空文本块跳过

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_elements.py -v`
Expected: PASS（8 passed，含 Task 2 的 3 个）

- [ ] **Step 5: Run the whole suite**

Run: `py -m pytest -q`
Expected: `137 passed`

- [ ] **Step 6: Commit**

```bash
git add ragv1/ingest/elements.py tests/test_elements.py
git commit -m "feat(ingest): Element 流分块，表格独立成块且每块带表头"
```

---

### Task 6: FtsStore 元数据列、schema 守卫与 `meta_of`

**Files:**
- Modify: `ragv1/store/fts_store.py`
- Test: `tests/test_meta.py`

**Interfaces:**
- Consumes: `ragv1.types.Chunk`
- Produces:
  - `FtsStore.SCHEMA_VERSION: int = 2`
  - `FtsStore.meta_of(chunk_id: str) -> dict | None`，键固定为 `doc_id, kind, page, order, part, bbox, image_ref, image_path, table_structured, degrade`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_meta.py
import json

from ragv1.store.fts_store import FtsStore
from ragv1.types import Chunk


def _chunk(**kw):
    base = dict(chunk_id="c1", doc_id="d.md", heading_path=("H",), text="正文")
    base.update(kw)
    return Chunk(**base)


def test_meta_of_returns_all_keys(tmp_path):
    store = FtsStore(tmp_path / "kb.db")
    store.add([_chunk(kind="table", page=3, order=7, part=1,
                      bbox=(1.0, 2.0, 3.0, 4.0),
                      image_ref="http://x/y.png", image_path="/tmp/y.png",
                      table_structured=False, degrade=("table_unstructured",))])
    meta = store.meta_of("c1")
    assert set(meta) == {
        "doc_id", "kind", "page", "order", "part", "bbox", "image_ref",
        "image_path", "table_structured", "degrade",
    }
    assert meta["doc_id"] == "d.md"
    assert meta["kind"] == "table"
    assert meta["page"] == 3 and meta["order"] == 7 and meta["part"] == 1
    assert meta["bbox"] == [1.0, 2.0, 3.0, 4.0]        # 已反序列化
    assert meta["degrade"] == ["table_unstructured"]   # 已反序列化
    assert meta["table_structured"] is False


def test_meta_of_defaults_for_plain_text(tmp_path):
    store = FtsStore(tmp_path / "kb.db")
    store.add([_chunk()])
    meta = store.meta_of("c1")
    assert meta["kind"] == "text" and meta["page"] is None
    assert meta["bbox"] is None and meta["degrade"] == []
    assert meta["table_structured"] is True


def test_meta_of_unknown_chunk_returns_none(tmp_path):
    assert FtsStore(tmp_path / "kb.db").meta_of("nope") is None


def test_old_schema_is_dropped_and_recreated(tmp_path):
    """版本不符时整表重建——索引是派生数据，全量重建无损失"""
    db = tmp_path / "kb.db"
    FtsStore(db).add([_chunk()])
    # 伪造旧版本
    import sqlite3
    con = sqlite3.connect(db)
    con.execute("UPDATE schema_meta SET value = '1' WHERE key = 'schema_version'")
    con.commit(); con.close()

    assert FtsStore(db).meta_of("c1") is None            # 旧数据被清掉
    assert FtsStore(db).chunk_ids() == set()


def test_repeated_add_does_not_duplicate_fts_rows(tmp_path):
    """重跑入库不能产生重复行 —— 重复块会让 RRF 给同一块加两次分"""
    store = FtsStore(tmp_path / "kb.db")
    store.add([_chunk()])
    store.add([_chunk()])
    assert store.search("正文", k=10).__len__() == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_meta.py -v`
Expected: FAIL — `AttributeError: 'FtsStore' object has no attribute 'meta_of'`

- [ ] **Step 3: Implement in `ragv1/store/fts_store.py`**

要点：

- 新增 `schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)` 表，初始化时读 `schema_version`；**缺失或 ≠ `SCHEMA_VERSION`（=2）**则 `DROP TABLE IF EXISTS chunks` / `DROP TABLE IF EXISTS chunks_fts`，重建后写入版本号。注释写明理由：索引是**派生的、永远全量重建**的数据，丢弃无损失
- `chunks` 表新增列：`kind TEXT NOT NULL DEFAULT 'text'`、`page INTEGER`、`"order" INTEGER NOT NULL DEFAULT 0`、`part INTEGER NOT NULL DEFAULT 0`、`bbox TEXT`、`image_ref TEXT`、`image_path TEXT`、`table_structured INTEGER NOT NULL DEFAULT 1`、`degrade TEXT NOT NULL DEFAULT '[]'`
  - ⚠️ **`order` 是 SQL 关键字，列名必须加双引号**，读写两处的 SQL 都要
- `add()` 写入新列：`bbox` → `json.dumps(list(c.bbox))` 或 `None`；`degrade` → `json.dumps(list(c.degrade))`；`table_structured` → `int(c.table_structured)`
- `meta_of()`：查 `chunks` 表，把 `bbox` / `degrade` 反序列化回 `list`（NULL 保持 `None` / `[]`），`table_structured` 转回 `bool`；查不到返回 `None`
- `search()` / `text_of()` / `chunk_ids()` 逻辑不变

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_meta.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: Run the whole suite**

Run: `py -m pytest -q`
Expected: `142 passed`（若 `tests/test_chinese.py` 等因重建逻辑失败，说明 schema 守卫误伤——修守卫，别改那些测试）

- [ ] **Step 6: Commit**

```bash
git add ragv1/store/fts_store.py tests/test_meta.py
git commit -m "feat(store): FTS 表增元数据列、schema 版本守卫与 meta_of 回源"
```

---

### Task 7: VectorStore 写入 Chroma metadata

**Files:**
- Modify: `ragv1/store/vector_store.py`
- Test: `tests/test_meta.py`（追加）

**Interfaces:**
- Consumes: `ragv1.types.Chunk`
- Produces: `ragv1.store.vector_store.chunk_metadata(c: Chunk) -> dict`（模块级函数，供测试直接断言）

- [ ] **Step 1: Write the failing test**

```python
# 追加到 tests/test_meta.py
from ragv1.store.vector_store import chunk_metadata


def test_chunk_metadata_only_uses_chroma_scalar_types():
    """Chroma 的 metadata 只接受 str/int/float/bool —— 元组和 None 都不行"""
    md = chunk_metadata(_chunk(kind="image", page=2, order=5, part=0,
                               bbox=(1.0, 2.0, 3.0, 4.0),
                               image_ref="http://x/y.png", image_path=None,
                               table_structured=True, degrade=("ocr_empty",)))
    for k, v in md.items():
        assert isinstance(v, (str, int, float, bool)), f"{k} 的类型 {type(v)} 不被 Chroma 接受"
    assert md["degrade"] == json.dumps(["ocr_empty"])     # 元组 → JSON 字符串
    assert md["bbox"] == json.dumps([1.0, 2.0, 3.0, 4.0])
    assert md["kind"] == "image" and md["page"] == 2
    assert "image_path" not in md                         # None 的键直接省略


def test_chunk_metadata_omits_all_none():
    md = chunk_metadata(_chunk())
    assert md == {"doc_id": "d.md", "kind": "text", "order": 0, "part": 0,
                  "table_structured": True, "degrade": "[]"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_meta.py -v`
Expected: FAIL — `ImportError: cannot import name 'chunk_metadata'`

- [ ] **Step 3: Implement in `ragv1/store/vector_store.py`**

- 新增模块级 `chunk_metadata(c: Chunk) -> dict`：值为 `None` 的键**省略**；`bbox` / `degrade` 用 `json.dumps(list(...))`；`table_structured` 用 `bool`；其余原样
- `add()` 增加 `metadatas=[chunk_metadata(c) for c in chunks]`
- `search()` 不动（本阶段不在 `Hit` 上带元数据；端到端暴露走 Task 8 的 `meta_lookup`）

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_meta.py -v`
Expected: PASS（7 passed）

- [ ] **Step 5: Run the whole suite**

Run: `py -m pytest -q`
Expected: `144 passed`

- [ ] **Step 6: Commit**

```bash
git add ragv1/store/vector_store.py tests/test_meta.py
git commit -m "feat(store): 向量库写入 Chroma metadatas（仅标量类型）"
```

---

### Task 8: `build.py` 接入新链路 + 集成测试

**Files:**
- Modify: `ragv1/ingest/build.py`
- Test: `tests/test_build_multimodal.py`
- Create: `tests/fixtures/table_doc.md`

**Interfaces:**
- Consumes: `ragv1.ingest.loader.load_document`、`ragv1.ingest.elements.chunk_elements`、`FtsStore.meta_of`
- Produces: `build_corpus(...)` 签名不变，内部改走新链路

- [ ] **Step 1: Create the fixture `tests/fixtures/table_doc.md`**

内容需含：一个正常段落、一张小表（能被整块容纳）、一段正文、一张**超过 1200 字符的大表**（用脚本生成 ≥30 行）、结尾段落。大表用来验证「不被字符硬切」。同时确保**不含图片引用**（本阶段不处理图片）。

- [ ] **Step 2: Write the failing test**

```python
# tests/test_build_multimodal.py
from pathlib import Path

from ragv1.ingest.build import build_corpus
from ragv1.store.fts_store import FtsStore

FIXTURES = Path(__file__).parent / "fixtures"


def test_build_indexes_table_doc_end_to_end(tmp_path):
    chunks = build_corpus(FIXTURES, tmp_path / "idx", embed_fn=lambda t: [0.0] * 8)
    assert chunks > 0

    store = FtsStore(tmp_path / "idx" / "kb.db")
    table_ids = [p for p in store.chunk_ids()
                 if store.meta_of(p)["kind"] == "table"]
    assert table_ids, "含表文档没有产出任何 table 类型的块"

    # 大表被切成多块，且每块都带表头
    pairs = [(cid, store.meta_of(cid)) for cid in table_ids]
    assert any(m["part"] > 0 for _, m in pairs), "大表没有被切分"
    for cid, _ in pairs:
        assert "| --- |" in store.text_of(cid).splitlines()[1]


def test_build_is_deterministic(tmp_path):
    a = build_corpus(FIXTURES, tmp_path / "a", embed_fn=lambda t: [0.0] * 8)
    b = build_corpus(FIXTURES, tmp_path / "b", embed_fn=lambda t: [0.0] * 8)
    sa = FtsStore(tmp_path / "a" / "kb.db")
    sb = FtsStore(tmp_path / "b" / "kb.db")
    assert a == b
    assert sa.chunk_ids() == sb.chunk_ids()


def test_build_skips_unknown_suffixes(tmp_path):
    """fixtures 里若有非 md 文件，不应让入库崩溃"""
    assert build_corpus(FIXTURES, tmp_path / "idx", embed_fn=lambda t: [0.0] * 8) > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `py -m pytest tests/test_build_multimodal.py -v`
Expected: FAIL — `build_corpus` 仍走旧 `parse_document`，不产出 `kind="table"` 的块

- [ ] **Step 4: Implement in `ragv1/ingest/build.py`**

把循环体从：

```python
doc = parse_document(md.read_text(encoding="utf-8"), doc_id=doc_id)
all_chunks.extend(chunk_document(doc, max_chars=MAX_CHARS))
```

改为：

```python
elements = load_document(md, doc_id)
all_chunks.extend(chunk_elements(elements, doc_id, MAX_CHARS))
```

`rglob` 的匹配范围从 `"*.md"` 放宽到 `"*"`，由 `load_document` 按后缀分派（未知后缀返回 `[]`）——**这是阶段二的 PDF 入口预留**。保持 `sorted()` 保证顺序确定。更新模块 docstring，说明数据流已变成 `load_document → chunk_elements`。

- [ ] **Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_build_multimodal.py -v`
Expected: PASS（3 passed）

- [ ] **Step 6: Run the whole suite**

Run: `py -m pytest -q`
Expected: `147 passed`

- [ ] **Step 7: Commit**

```bash
git add ragv1/ingest/build.py tests/test_build_multimodal.py tests/fixtures/table_doc.md
git commit -m "feat(ingest): build_corpus 接入 Element 链路，支持表格结构化入库"
```

---

### Task 9: API 端到端暴露来源元数据

**Files:**
- Modify: `ragv1/api/main.py`
- Test: `tests/test_api.py`（追加）

**Interfaces:**
- Consumes: `FtsStore.meta_of`
- Produces: `create_app(retriever, meta_lookup: Callable[[str], dict | None] | None = None) -> FastAPI`；`ResultItem` 新增 `doc_id, kind, page, order, bbox, image_ref, image_path, table_structured, degrade`（全部可选带默认值）

**设计取舍**：不把 store 塞进 `rrf_fuse`——融合层的契约是「只做排名、不认识 store」（见 `ragv1/fusion/rrf.py` 文档）。查表在 API 边界注入。

- [ ] **Step 1: Write the failing test**

```python
# 追加到 tests/test_api.py
from fastapi.testclient import TestClient
from ragv1.api.main import create_app


def test_metadata_is_absent_when_lookup_not_supplied(monkeypatch):
    """缺省 None 时行为与改造前完全一致 —— 现有 API 测试就是这条的证据"""
    app = create_app(_stub_retriever())
    r = TestClient(app).post("/search", json={"query": "q", "k": 3})
    assert r.status_code == 200
    assert "doc_id" not in r.json()["results"][0]


def test_metadata_is_returned_when_lookup_supplied():
    def lookup(chunk_id):
        return {"doc_id": "d.md", "kind": "table", "page": 3, "order": 7,
                "part": 1, "bbox": [1.0, 2.0, 3.0, 4.0],
                "image_ref": "http://x/y.png", "image_path": None,
                "table_structured": False, "degrade": ["table_unstructured"]}

    app = create_app(_stub_retriever(), meta_lookup=lookup)
    item = TestClient(app).post("/search", json={"query": "q", "k": 3}).json()["results"][0]
    assert item["doc_id"] == "d.md" and item["kind"] == "table"
    assert item["page"] == 3 and item["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert item["degrade"] == ["table_unstructured"]
    assert item["table_structured"] is False


def test_lookup_returning_none_does_not_crash():
    """chunk 不在回源表里（例如索引与元数据不同步）时不能 500"""
    app = create_app(_stub_retriever(), meta_lookup=lambda cid: None)
    r = TestClient(app).post("/search", json={"query": "q", "k": 3})
    assert r.status_code == 200
```

`_stub_retriever()` 复用 `tests/test_api.py` 里已有的假 retriever 构造方式（若没有则按 `tests/test_degradation.py` 的 `Ok` 类新建一个）。

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_api.py -v`
Expected: FAIL — `TypeError: create_app() got an unexpected keyword argument 'meta_lookup'`

- [ ] **Step 3: Implement in `ragv1/api/main.py`**

- `create_app(retriever, meta_lookup=None)`；`meta_lookup` 为 `None` 时保持现状
- `ResultItem` 追加九个可选字段（默认 `None` / `False` / `[]`）
- 构造 `ResultItem` 前：`meta = meta_lookup(h.chunk_id) if meta_lookup else None`，然后用 `**(meta or {})` 展开——**必须处理返回 `None` 的情况**
- 模块 docstring 补一句：来源元数据由注入的 `meta_lookup`（生产为 `FtsStore.meta_of`）提供，融合层保持对 store 无感知
- **生产接线（本仓库目前没有装配入口）**：仓库里 `create_app` 只在测试中被构造，没有 `main.py` 之类的启动脚本。因此把用法写进 `ragv1/api/main.py` 的模块 docstring 顶部，三行示例：

  ```python
  # 生产用法（需先 build_corpus 建好索引）：
  #   store = FtsStore(INDEX_DIR / "kb.db")
  #   app = create_app(retriever, meta_lookup=store.meta_of)
  ```

  不要为此新建启动脚本——spec 的 Out-of-scope 是「无部署」，凭空加一个入口属于超范围。

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Run the whole suite**

Run: `py -m pytest -q`
Expected: `150 passed`

- [ ] **Step 6: Commit**

```bash
git add ragv1/api/main.py tests/test_api.py
git commit -m "feat(api): 检索结果带出来源元数据（meta_lookup 注入）"
```

---

### Task 10: 阶段一验收——用真实语料量一次

**Files:**
- Create: `scripts/verify_tables.py`

**Interfaces:**
- Consumes: `build_corpus`、`FtsStore`
- Produces: 一份**实测数字**，写进 README 与提交信息（不是"跑通了"这种空话）

- [ ] **Step 1: Write `scripts/verify_tables.py`**

对 `config.CORPUS_DIR` 真跑一遍 `build_corpus`，输出：

- 总块数、`kind` 分布（text / table）
- 表格块总数、其中 `part > 0` 的块数（即被切分的大表）
- `degrade` 非空的块数及原因码分布
- **最长表格块字符数**（应 > `MAX_CHARS`，证明超限行没被切）
- 对照 spec §1 的基线（152 张表 / 9 张超限），打印前后对比

脚本用 `embed_fn=lambda t: [0.0]*8` 假向量（不调付费 API）。

- [ ] **Step 2: Run it**

Run: `py scripts/verify_tables.py`
Expected: 打印上述数字。**把实际输出记下来**——它是阶段一的验收证据

- [ ] **Step 3: 核对验收标准**

- [ ] 表格块数 ≥ 152（语料基线），且每块都带表头
- [ ] 最长表格块 > 1200 字符（说明超限行未被切）
- [ ] 无任何块因表格被 `_pieces_within` 硬切
- [ ] `py -m pytest -q` 全绿，且数量 > 150

- [ ] **Step 4: 更新 README**

在「实测数字」表里加一行阶段一的实测结果（真实数字，不是占位）。

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_tables.py README.md
git commit -m "chore: 阶段一验收脚本与实测数字"
```

---

## 阶段一完成后的状态

- 表格独立成块、结构化、每块带表头、超限行不被切
- 每个 chunk 带完整来源元数据，API 可带出
- 纯文本文档产出与改造前逐字节一致（有回归测试为证）
- **零新依赖**，`requirements.txt` 未动
- 图片仍按今天的方式处理（`alt` 文字留在正文）——交给阶段二
