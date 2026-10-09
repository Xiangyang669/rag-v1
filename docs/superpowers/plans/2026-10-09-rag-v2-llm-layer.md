# rag-v2 LLM 层 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 rag-v1 的三层检索内核补上生成层（引用溯源 + 拒答 + 多轮），并先把评估集换成诚实的、可支撑结论的那一套。

**Architecture:** 沿用现有 LangGraph Supervisor（`orchestration/supervisor.py`），在图前端加 `rewrite` 节点、后端加 `generate` / `verify` 节点。**节点是否存在于图中由构造期参数决定**（`llm=None` → 没有这三个节点 → 就是 V1 的检索图），沿用 `build_graph` 对 `paths` 已有的同一套做法。融合层由等权 RRF 改为加权 RRF + 查询自适应权重。

**Tech Stack:** Python 3 · LangGraph · FastAPI · SQLite FTS5 · Chroma · SiliconFlow（embedding + LLM，OpenAI 兼容）· pytest

**Spec:** [`docs/superpowers/specs/2026-10-09-rag-v2-llm-layer-design.md`](../specs/2026-10-09-rag-v2-llm-layer-design.md)

## Global Constraints

- **零回归**：231 个 V1 测试必须全绿（`py -m pytest -q`）。任何一条变红都是阻塞。
- **`rrf_fuse(ranked)` 不传 `weights` 时，输出与 V1 逐字节一致**。这是零回归的前提。
- **`/search` 端点的请求/响应行为一字不改**。
- **可复现**：同一输入两次运行结果完全一致；一切排序平局按 `chunk_id` 升序打破。
- **离线可测**：所有外部依赖（embedding、LLM）必须可注入假实现；无 API key 时 `py -m pytest` 仍须全绿。
- **降级不静默**：解析失败落降级码（`ingest/degrade.py` 同款思路），不吞掉、不伪造。
- **凭据不硬编码**：走 `embedding.resolve_api_key` 已有的解析顺序（环境变量 → 项目 `.env` → 上层 `practice/.env`）。
- **中文分词只作用于 `chunks_fts`**；`chunks.text` 永远保持原文（见 `store/fts_store.py:7-8`）。
- **bm25 是负值，越小越相关**（`store/fts_store.py:10`）。任何读 `Hit.score` 的新代码都必须先统一方向。

## Review Focus

以下五类是 spec 隐含、但若不专门写测试就不会被覆盖的失败模式。每一条都已在对应任务里落成测试。

1. **bm25 方向读反** —— 路由若把"最相关"当成"最不相关"，权重会朝相反方向调，且**测试之外的任何指标都不会报错**，只会让融合悄悄变差。（Task 6）
2. **拒答只做了一层** —— 上下文非空但与问题无关时，只看融合分会硬答、只看模型会在空上下文上编答案。（Task 11）
3. **引用标记与 citations 数组不一致** —— 模型在正文写了 `[3]` 但返回的 `citations` 只有 `[1,2]`，或反之。（Task 12）
4. **`/ask` 在 LLM 未配置时的行为** —— 必须明确报错，**绝不能静默降级成"只返回检索结果"**，那会让调用方拿到一个看起来正常、实际没有答案的响应。（Task 14）
5. **双路检索中一条变体失败** —— 改写查询触发的那路 embedding 失败时，不能拖垮整次查询，也不能让原始查询那一路被误标为降级。（Task 17）

---

## 文件结构

**新建：**

| 文件 | 职责 |
|---|---|
| `ragv1/fusion/router.py` | 查询自适应权重（纯规则、零 LLM、确定性） |
| `ragv1/qa/__init__.py` | 包入口 |
| `ragv1/qa/llm.py` | LLM 客户端协议 + SiliconFlow 实现 + 可注入假客户端 |
| `ragv1/qa/prompt.py` | 编号上下文渲染 + 系统提示词 + 响应解析 |
| `ragv1/qa/generate.py` | 生成带引用的答案 + 两层拒答 |
| `ragv1/qa/verify.py` | 机械引用校验 + 可选支持度判定 |
| `ragv1/qa/rewrite.py` | 多轮指代消解 |
| `ragv1/evaluation/qaset.py` | 人工评估集的类型、加载、格式校验 |
| `ragv1/evaluation/answer_metrics.py` | faithfulness / citation_accuracy / refusal_* |
| `ragv1/evaluation/qa_sets/zh_dev.jsonl` | 人工评估集 dev（用于调参） |
| `ragv1/evaluation/qa_sets/zh_test.jsonl` | 人工评估集 test（**调参时不得触碰**） |
| `scripts/find_chunks.py` | 标注辅助：按查询打印 chunk_id，供人标答案块 |
| `scripts/tune_weights.py` | dev 集网格搜索静态融合权重 |
| `scripts/eval_v2.py` | 跑完整验收矩阵，产出报告数字 |

**修改：**

| 文件 | 改动 |
|---|---|
| `ragv1/config.py` | 新增融合权重、改写折扣、拒答阈值、QA top-N、LLM 配置 |
| `ragv1/fusion/rrf.py` | `rrf_fuse` 加 `weights` 参数；`FusedHit` 加 `contributions` |
| `ragv1/types.py` | `FusedHit.contributions`；新增 `Turn` / `Answer` / `VerifyReport` |
| `ragv1/orchestration/supervisor.py` | State 扩字段；`_fuse_node` 接路由；加 `rewrite`/`generate`/`verify` 节点；`run_ask_meta` |
| `ragv1/api/main.py` | 新增 `/ask`；`build_graph` 调用点传 `text_lookup` / `llm` |
| `ragv1/api/server.py` | 装配 `text_lookup=FtsStore.text_of` 与 LLM 客户端 |
| `README.md` | V2 取舍表新增行 |

---

# 里程碑 M1 · 真评估（前置）

> 没有诚实的评估集，M2–M4 的每一个数字都是自欺。

### Task 1: 中文语料入库

**Files:**
- Modify: `ragv1/config.py`（新增中文语料与索引目录常量）
- Test: `tests/test_zh_corpus_config.py`

**Interfaces:**
- Consumes: `scripts/build_index.py` 的 `--corpus` / `--index-dir`（已存在，不改）
- Produces: `config.ZH_CORPUS_DIR: Path`、`config.ZH_INDEX_DIR: Path`（= `.indexes/kb_zh`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_zh_corpus_config.py
def test_zh_index_dir_is_separate_from_en_index():
    from ragv1 import config
    assert config.ZH_INDEX_DIR != config.INDEX_DIR / "kb"
    assert config.ZH_INDEX_DIR.name == "kb_zh"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_zh_corpus_config.py -v`
Expected: FAIL — `AttributeError: module 'ragv1.config' has no attribute 'ZH_INDEX_DIR'`

- [ ] **Step 3: 在 `ragv1/config.py` 新增常量**

```python
# V2 中文主语料：与英文基准分属两套索引，互不覆盖
ZH_CORPUS_DIR = Path(os.environ.get("RAGV1_ZH_CORPUS", "D:/corpus/dify-docs"))
ZH_INDEX_DIR = INDEX_DIR / "kb_zh"
```

- [ ] **Step 4: 跑测试确认通过**

Run: `py -m pytest tests/test_zh_corpus_config.py -v`
Expected: PASS

- [ ] **Step 5: 落地语料（人工步骤）**

下载 Dify 或 FastGPT 的中文文档仓库，解压到 `ZH_CORPUS_DIR`。
（若机器上已有更合适的中文技术文档，改 `RAGV1_ZH_CORPUS` 环境变量指向它。）

- [ ] **Step 6: 建中文索引并验证可查**

Run: `py scripts/build_index.py --corpus "<ZH_CORPUS_DIR>" --index-dir .indexes/kb_zh`
Expected: 打印块数 > 0、无异常退出。

Run: `py scripts/verify_ingest.py --index-dir .indexes/kb_zh`
Expected: 表格行完整性检查通过（每个表格块每行以 `|` 收尾）。

- [ ] **Step 7: 确认英文基准未受影响**

Run: `py -m pytest -q`
Expected: 231 passed

- [ ] **Step 8: Commit**

```bash
git add ragv1/config.py tests/test_zh_corpus_config.py
git commit -m "feat(config): 中文主语料与独立索引目录"
```

---

### Task 2: 评估集数据结构与加载器

**Files:**
- Create: `ragv1/evaluation/qaset.py`
- Test: `tests/test_qaset.py`

**Interfaces:**
- Consumes: `ragv1.types.Turn`（Task 16 定义；**本任务先在同文件内联一个等价的 `Turn` dataclass 会重复**——改为本任务在 `ragv1/types.py` 中先加 `Turn`，Task 16 直接复用）
- Produces:
  - `types.Turn(role: str, text: str)`
  - `qaset.EvalItem(question: str, answer_chunk_ids: tuple[str, ...], category: str, history: tuple[Turn, ...] = ())`
  - `qaset.load_qaset(path: str | Path) -> tuple[EvalItem, ...]`
  - `qaset.QASetError`（格式错误时抛出）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qaset.py
import json, pytest
from ragv1.evaluation.qaset import load_qaset, QASetError

def _write(tmp_path, rows):
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return p

def test_loads_valid_rows(tmp_path):
    p = _write(tmp_path, [{"question": "怎么配？", "answer_chunk_ids": ["a", "b"], "category": "multi_hop"}])
    items = load_qaset(p)
    assert len(items) == 1 and items[0].answer_chunk_ids == ("a", "b")

def test_row_without_history_gets_empty_tuple(tmp_path):
    p = _write(tmp_path, [{"question": "q", "answer_chunk_ids": ["a"], "category": "fact"}])
    assert load_qaset(p)[0].history == ()

def test_unanswerable_row_may_have_empty_answer(tmp_path):
    p = _write(tmp_path, [{"question": "库里没有的", "answer_chunk_ids": [], "category": "unanswerable"}])
    assert load_qaset(p)[0].answer_chunk_ids == ()

def test_rejects_missing_question(tmp_path):
    p = _write(tmp_path, [{"answer_chunk_ids": ["a"], "category": "fact"}])
    with pytest.raises(QASetError, match="question"):
        load_qaset(p)

def test_rejects_blank_line_noise(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"question":"q","answer_chunk_ids":["a"],"category":"fact"}\n\n', encoding="utf-8")
    assert len(load_qaset(p)) == 1

def test_reports_line_number_on_bad_json(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"question":"q","answer_chunk_ids":["a"],"category":"fact"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(QASetError, match="第 2 行"):
        load_qaset(p)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_qaset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragv1.evaluation.qaset'`

- [ ] **Step 3: 在 `ragv1/types.py` 加 `Turn`，并实现 `qaset.py`**

`types.py` 追加（`Answer` / `VerifyReport` 在 Task 11 / 12 加，本轮不加）：

```python
@dataclass(frozen=True)
class Turn:
    role: str   # "user" | "assistant"
    text: str
```

`qaset.py` 用 `json` 逐行解析；空白行跳过；`history` 每一元素转成 `Turn`；
缺 `question` / `answer_chunk_ids` / `category` 任一 → `QASetError(f"第 {n} 行：缺少 {field}")`；
JSON 解析失败 → `QASetError(f"第 {n} 行：{err}")`。

- [ ] **Step 4: 跑测试确认通过**

Run: `py -m pytest tests/test_qaset.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ragv1/types.py ragv1/evaluation/qaset.py tests/test_qaset.py
git commit -m "feat(evaluation): 人工评估集的数据结构与加载器"
```

---

### Task 3: 标注辅助脚本

**Files:**
- Create: `scripts/find_chunks.py`
- Test: `tests/test_find_chunks.py`

**Interfaces:**
- Consumes: `api.server.build_retriever(idx, embed_fn)`、`store.FtsStore.text_of` / `meta_of`
- Produces: 命令行 `py scripts/find_chunks.py "查询" [--index-dir ...] [--k 10]`，
  打印 `chunk_id` · `doc_id` · `page` · 前 80 字原文

- [ ] **Step 1: 写失败测试**

```python
# tests/test_find_chunks.py
import importlib.util, sys
from pathlib import Path

def _load():
    spec = importlib.util.spec_from_file_location(
        "find_chunks", Path(__file__).resolve().parents[1] / "scripts" / "find_chunks.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_render_line_contains_id_and_snippet():
    mod = _load()
    line = mod.render_row("cid1", {"doc_id": "d", "page": 3}, "很长的正文" * 40)
    assert "cid1" in line and "d" in line and "3" in line
    assert len(line) < 400          # 摘要必须被截断

def test_render_row_tolerates_missing_meta():
    mod = _load()
    assert "cid1" in mod.render_row("cid1", None, None)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_find_chunks.py -v`
Expected: FAIL — 文件不存在

- [ ] **Step 3: 实现 `scripts/find_chunks.py`**

`render_row(chunk_id: str, meta: dict | None, text: str | None) -> str`
——纯函数，便于测试；`main()` 装配 retriever 后对每个融合命中调用它并打印。
`meta` 为 `None` 时不崩，原文截断到 80 字。

- [ ] **Step 4: 跑测试确认通过**

Run: `py -m pytest tests/test_find_chunks.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/find_chunks.py tests/test_find_chunks.py
git commit -m "feat(scripts): 标注辅助脚本 find_chunks"
```

---

### Task 4: 撰写人工评估集（内容任务，非代码）

**Files:**
- Create: `ragv1/evaluation/qa_sets/zh_dev.jsonl`
- Create: `ragv1/evaluation/qa_sets/zh_test.jsonl`
- Test: `tests/test_qasets_present.py`

**Interfaces:**
- Consumes: `qaset.load_qaset`（Task 2）
- Produces: 两个 JSONL 文件，行格式 `{"question", "answer_chunk_ids", "category", "history"?}`

- [ ] **Step 1: 写失败测试（把评估集的构成要求钉死）**

```python
# tests/test_qasets_present.py
import pytest
from pathlib import Path
from ragv1.evaluation.qaset import load_qaset

SETS = Path(__file__).resolve().parents[1] / "ragv1" / "evaluation" / "qa_sets"

@pytest.mark.parametrize("name", ["zh_dev.jsonl", "zh_test.jsonl"])
def test_set_meets_composition_requirements(name):
    items = load_qaset(SETS / name)
    cats = {i.category for i in items}
    assert len(items) >= 15, f"{name} 条数不足"
    assert "unanswerable" in cats,          f"{name} 缺不可回答题（测拒答必需）"
    assert cats & {"multi_hop", "comparison", "temporal"}, f"{name} 缺多跳/比较/时序题"

def test_dev_and_test_do_not_overlap():
    dev = {i.question for i in load_qaset(SETS / "zh_dev.jsonl")}
    test = {i.question for i in load_qaset(SETS / "zh_test.jsonl")}
    assert not (dev & test), "dev/test 有重复问题，调参会泄漏"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_qasets_present.py -v`
Expected: FAIL — 文件不存在

- [ ] **Step 3: 标注（人工步骤，用 Task 3 的脚本取 chunk_id）**

用 `py scripts/find_chunks.py "你的问题" --index-dir .indexes/kb_zh` 找到答案块，记下 `chunk_id`。
分类取值：`fact` / `multi_hop` / `comparison` / `temporal` / `unanswerable` / `multi_turn`。

**先写 10 条 dev 跑通全流程，再扩到 dev / test 各 ≥20 条。**「不可回答」类要有 3 条以上。

- [ ] **Step 4: 跑测试确认通过**

Run: `py -m pytest tests/test_qasets_present.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ragv1/evaluation/qa_sets tests/test_qasets_present.py
git commit -m "feat(evaluation): 人工评估集 zh_dev / zh_test"
```

---

# 里程碑 M2 · 加权融合（修 V1 未通过的判据）

### Task 5: 加权 RRF

**Files:**
- Modify: `ragv1/fusion/rrf.py`
- Modify: `ragv1/types.py`（`FusedHit.contributions`）
- Modify: `ragv1/config.py`
- Test: `tests/test_rrf.py`（追加）

**Interfaces:**
- Consumes: `types.Hit`、`config.RRF_K`
- Produces:
  - `rrf_fuse(ranked: dict[str, list[Hit]], k: int = RRF_K, weights: dict[str, float] | None = None) -> list[FusedHit]`
  - `FusedHit.contributions: dict[str, float]`（默认空 dict）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_rrf.py 追加
from ragv1.fusion.rrf import rrf_fuse
from ragv1.types import Hit

def _h(cid, rank, path, score=0.0):
    return Hit(chunk_id=cid, rank=rank, score=score, path=path)

def test_no_weights_is_byte_identical_to_v1():
    ranked = {"vector": [_h("a", 1, "vector"), _h("b", 2, "vector")],
              "fulltext": [_h("b", 1, "fulltext"), _h("c", 2, "fulltext")]}
    legacy = rrf_fuse(dict(ranked))
    explicit_none = rrf_fuse(dict(ranked), weights=None)
    assert [h.chunk_id for h in legacy] == [h.chunk_id for h in explicit_none]

def test_zero_weight_path_is_excluded():
    ranked = {"vector": [_h("a", 1, "vector")], "graph": [_h("z", 1, "graph")]}
    ids = [h.chunk_id for h in rrf_fuse(ranked, weights={"vector": 1.0, "graph": 0.0})]
    assert ids == ["a"]

def test_weight_scales_contribution():
    ranked = {"vector": [_h("a", 1, "vector")], "graph": [_h("z", 1, "graph")]}
    top = rrf_fuse(ranked, weights={"vector": 0.9, "graph": 0.1})[0]
    assert top.chunk_id == "a"

def test_unknown_weight_key_is_ignored():
    ranked = {"vector": [_h("a", 1, "vector")]}
    assert rrf_fuse(ranked, weights={"nosuch": 99.0})[0].chunk_id == "a"

def test_path_missing_from_weights_defaults_to_one():
    ranked = {"vector": [_h("a", 1, "vector")], "graph": [_h("z", 1, "graph")]}
    fused = rrf_fuse(ranked, weights={"vector": 1.0})
    assert {h.chunk_id for h in fused} == {"a", "z"}

def test_contributions_records_per_path_score():
    ranked = {"vector": [_h("a", 1, "vector")]}
    top = rrf_fuse(ranked, k=60, weights={"vector": 2.0})[0]
    assert top.contributions == {"vector": 2.0 / 61}

def test_tie_break_is_still_chunk_id_ascending():
    ranked = {"vector": [_h("b", 1, "vector")], "graph": [_h("a", 1, "graph")]}
    assert [h.chunk_id for h in rrf_fuse(ranked)] == ["a", "b"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_rrf.py -v -k "weight or contribution"`
Expected: FAIL — `TypeError: rrf_fuse() got an unexpected keyword argument 'weights'`

- [ ] **Step 3: 实现**

`FusedHit` 加 `contributions: dict[str, float] = field(default_factory=dict)`（需 `from dataclasses import field`）。
`rrf_fuse` 内：`w = 1.0 if weights is None else weights.get(path, 1.0)`；`w <= 0` → `continue`；
累加 `w / (k + hit.rank)` 进 `scores`，同时写进该块的 `contributions[path]`。排序与平局规则**保持原样**。

- [ ] **Step 4: 跑测试确认通过**

Run: `py -m pytest tests/test_rrf.py -v`
Expected: 全部 PASS（含 V1 既有用例）

- [ ] **Step 5: 全量回归**

Run: `py -m pytest -q`
Expected: 231 + 新增 passed

- [ ] **Step 6: Commit**

```bash
git add ragv1/fusion/rrf.py ragv1/types.py ragv1/config.py tests/test_rrf.py
git commit -m "feat(fusion): 加权 RRF + 每路贡献分"
```

---

### Task 6: 查询自适应权重

**Files:**
- Create: `ragv1/fusion/router.py`
- Modify: `ragv1/config.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: `types.Hit`、`config.STATIC_FUSION_WEIGHTS`、`config.REWRITE_WEIGHT_DISCOUNT`
- Produces: `route_weights(query: str, ranked: dict[str, list[Hit]], static: dict[str, float] | None = None) -> dict[str, float]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_router.py
from ragv1.fusion.router import route_weights
from ragv1.types import Hit

def _h(cid, rank, path, score):
    return Hit(chunk_id=cid, rank=rank, score=score, path=path)

BASE = {"vector": 0.50, "fulltext": 0.35, "graph": 0.15}

def test_static_baseline_when_no_signal():
    w = route_weights("数据库连接怎么配", {}, static=dict(BASE))
    assert w["vector"] == 0.50

def test_identifier_query_upweights_fulltext():
    w = route_weights("MAX_EMBED_BATCH 是多少", {}, static=dict(BASE))
    assert w["fulltext"] > BASE["fulltext"]

def test_backticked_term_upweights_fulltext():
    w = route_weights("`resolve_api_key` 怎么用", {}, static=dict(BASE))
    assert w["fulltext"] > BASE["fulltext"]

# ⚠️ Review Focus 1：bm25 是负值，越小越相关。方向读反会让权重朝反方向调。
def test_dominant_fulltext_hit_upweights_fulltext_bm25_is_negative():
    ranked = {"fulltext": [_h("a", 1, "fulltext", -9.0),
                           _h("b", 2, "fulltext", -3.0),
                           _h("c", 3, "fulltext", -3.0),
                           _h("d", 4, "fulltext", -3.0)]}
    w = route_weights("随便的问题", ranked, static=dict(BASE))
    assert w["fulltext"] > BASE["fulltext"]

def test_flat_fulltext_scores_do_not_upweight():
    ranked = {"fulltext": [_h("a", 1, "fulltext", -3.0), _h("b", 2, "fulltext", -3.1)]}
    w = route_weights("随便的问题", ranked, static=dict(BASE))
    assert w["fulltext"] == BASE["fulltext"]

def test_rewritten_variant_is_discounted():
    w = route_weights("追问", {"vector": [_h("a", 1, "vector", 0.9)]}, static=dict(BASE))
    assert "vector#rw" not in w        # 只有变体存在时才产出它

def test_rewritten_variant_discount_applied():
    ranked = {"vector#rw": [_h("a", 1, "vector#rw", 0.9)]}
    w = route_weights("追问", ranked, static=dict(BASE))
    assert w["vector#rw"] < w["vector"]

def test_is_deterministic():
    ranked = {"fulltext": [_h("a", 1, "fulltext", -9.0)]}
    assert route_weights("q", ranked, static=dict(BASE)) == route_weights("q", ranked, static=dict(BASE))

def test_result_covers_every_path_present_in_ranked():
    ranked = {"graph": [_h("g", 1, "graph", 1.0)]}
    assert "graph" in route_weights("q", ranked, static=dict(BASE))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_router.py -v`
Expected: FAIL — 模块不存在

- [ ] **Step 3: 在 `ragv1/config.py` 加常量**

```python
# 静态融合权重初值：调研查到的企业三信号示例，**必须在 dev 集上重调**（Task 8）
STATIC_FUSION_WEIGHTS = {"vector": 0.50, "fulltext": 0.35, "graph": 0.15}
# 改写变体的折扣（双路检索时压低改写路，避免它带偏）
REWRITE_WEIGHT_DISCOUNT = 0.5
# 词面信号触发时，全文路的上调倍数 / 向量路的下调倍数
LEXICAL_FULLTEXT_BOOST = 1.8
LEXICAL_VECTOR_DAMP = 0.6
# 全文路「压倒性命中」的判据：median(scores) - top1 超过此值（bm25 负值，越大越相关）
FULLTEXT_DOMINANCE_GAP = 3.0
REWRITTEN_SUFFIX = "#rw"
```

- [ ] **Step 4: 实现 `route_weights`**

规则（全部确定性、零 LLM）：
1. 从 `static`（缺省取 `config.STATIC_FUSION_WEIGHTS`）的副本出发。
2. **词面信号**：query 含反引号包裹片段、或含下划线/驼峰/全大写缩写（≥2 字符）
   → `fulltext *= LEXICAL_FULLTEXT_BOOST`，`vector *= LEXICAL_VECTOR_DAMP`。
3. **全文路压倒性命中**：`ranked["fulltext"]` 有 ≥3 条时，
   `median(scores) - scores[0] > FULLTEXT_DOMINANCE_GAP` → 同信号 2 的调整
   （用 `statistics.median`）。
4. **改写变体**：对 `ranked` 中带 `REWRITTEN_SUFFIX` 的键，产出同名权重 =
   同路径权重 × `REWRITE_WEIGHT_DISCOUNT`。不在 `ranked` 里的变体键不产出。

- [ ] **Step 5: 跑测试确认通过**

Run: `py -m pytest tests/test_router.py -v`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add ragv1/fusion/router.py ragv1/config.py tests/test_router.py
git commit -m "feat(fusion): 查询自适应权重路由"
```

---

### Task 7: 路由接入融合节点

**Files:**
- Modify: `ragv1/orchestration/supervisor.py`
- Test: `tests/test_supervisor.py`（追加）

**Interfaces:**
- Consumes: `fusion.router.route_weights`、`fusion.rrf.rrf_fuse`
- Produces: `State` 新增 `weights: dict[str, float]`；`_fuse_node` 先算权重再融合

- [ ] **Step 1: 写失败测试**

```python
# tests/test_supervisor.py 追加
def test_fuse_node_records_route_weights(monkeypatch):
    """融合节点必须把路由权重写回 state —— 这是调参时唯一的观测点。"""
    from ragv1.orchestration import supervisor
    out = supervisor._fuse_node({
        "query": "MAX_EMBED_BATCH 是多少",
        "ranked": {"fulltext": [Hit(chunk_id="a", rank=1, score=-9.0, path="fulltext")],
                   "vector": [Hit(chunk_id="b", rank=1, score=0.9, path="vector")]},
        "on_error": "skip",
    })
    assert out["weights"]["fulltext"] > 0
    assert len(out["fused"]) == 2

def test_fuse_node_with_empty_ranked_returns_empty():
    from ragv1.orchestration import supervisor
    out = supervisor._fuse_node({"query": "q", "ranked": {}, "on_error": "skip"})
    assert out["fused"] == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_supervisor.py -v -k "route_weights or empty_ranked"`
Expected: FAIL — `KeyError: 'weights'`

- [ ] **Step 3: 改 `_fuse_node` 与 `State`**

`State` 加 `weights: dict[str, float]`。
`_fuse_node` 改成：`weights = route_weights(state["query"], state["ranked"])`，
返回 `{"fused": rrf_fuse(state["ranked"], weights=weights), "weights": weights}`。

- [ ] **Step 4: 跑测试确认通过并全量回归**

Run: `py -m pytest -q`
Expected: 全部 passed（V1 既有融合用例若断言了未加权结果而失败，**先看是不是真回归**再动）

- [ ] **Step 5: Commit**

```bash
git add ragv1/orchestration/supervisor.py tests/test_supervisor.py
git commit -m "feat(orchestration): 融合节点接入查询自适应权重"
```

---

### Task 8: 权重调参脚本与 dev 集定档

**Files:**
- Create: `scripts/tune_weights.py`
- Test: `tests/test_tune_weights.py`
- Modify: `ragv1/config.py`（把搜到的最优值写回 `STATIC_FUSION_WEIGHTS`）

**Interfaces:**
- Consumes: `api.server.build_retriever`、`evaluation.qaset.load_qaset`、`evaluation.harness` 的指标函数、`fusion.rrf.rrf_fuse`
- Produces:
  - `tune_weights.grid(k_values, grids) -> list[tuple[dict[str, float], dict[str, float]]]`（按 recall@5 降序）
  - `tune_weights.score_weights(retriever, items, weights, k) -> dict[str, float]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tune_weights.py
import importlib.util
from pathlib import Path

def _load():
    spec = importlib.util.spec_from_file_location(
        "tune_weights", Path(__file__).resolve().parents[1] / "scripts" / "tune_weights.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_grid_is_deterministic_and_descending():
    mod = _load()
    fake = [({"vector": 0.5}, {"recall@5": 0.4}), ({"vector": 0.9}, {"recall@5": 0.8})]
    out = mod.rank_results(fake)
    assert out[0][0] == {"vector": 0.9}

def test_grid_enumerates_all_combinations():
    mod = _load()
    combos = list(mod.weight_combos((0.3, 0.6), (0.7,)))
    assert len(combos) == 2 and all(abs(sum(c.values()) - 1.0) < 1e-9 for c in combos)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_tune_weights.py -v`
Expected: FAIL — 文件不存在

- [ ] **Step 3: 实现 `scripts/tune_weights.py`**

`weight_combos(*axes)` 生成归一化的权重字典（`graph` 权重 = 1 − vector − fulltext，
且必须 ≥ 0，否则跳过）；`score_weights` 对 dev 集算 recall@5 / MRR；
`rank_results` 按 recall@5 降序、权重字典的 `sorted(items())` 作平局序（保证可复现）。
`main()` 用真实索引跑，打印 Top-10 组合。

- [ ] **Step 4: 跑测试确认通过**

Run: `py -m pytest tests/test_tune_weights.py -v`
Expected: 2 passed

- [ ] **Step 5: 在 dev 集上定档（人工步骤）**

Run: `py scripts/tune_weights.py --index-dir .indexes/kb_zh --dev ragv1/evaluation/qa_sets/zh_dev.jsonl`
把最优权重写回 `config.STATIC_FUSION_WEIGHTS`。**记录搜索范围与最终取值。**

- [ ] **Step 6: 在 test 集上出判据 1 的数字**

Run: `py scripts/eval_v2.py --index-dir .indexes/kb_zh --set ragv1/evaluation/qa_sets/zh_test.jsonl --retrieval-only`
Expected: 打印加权融合 vs 各单路的照表。**判据 1 = 融合 recall@5 ≥ 最强单路。**

- [ ] **Step 7: Commit**

```bash
git add scripts/tune_weights.py tests/test_tune_weights.py ragv1/config.py
git commit -m "feat(evaluation): 静态权重网格搜索 + dev 集定档"
```

---

# 里程碑 M3 · 生成层

### Task 9: LLM 客户端

**Files:**
- Create: `ragv1/qa/__init__.py`、`ragv1/qa/llm.py`
- Modify: `ragv1/config.py`
- Test: `tests/test_qa_llm.py`

**Interfaces:**
- Consumes: `embedding.resolve_api_key`、`embedding.ENV_FILE_CANDIDATES`
- Produces:
  - `llm.LLMClient`（Protocol，方法 `chat(messages: list[dict], **kwargs) -> str`）
  - `llm.LLMNotConfigured(RuntimeError)`
  - `llm.SiliconFlowClient(model=None, timeout=None)`，构造时解析 key，缺 key 抛 `LLMNotConfigured`
  - `llm.FakeLLM(replies: list[str])`，测试用，记录 `calls: list[list[dict]]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qa_llm.py
import pytest
from ragv1.qa.llm import FakeLLM, SiliconFlowClient, LLMNotConfigured

def test_fake_llm_returns_queued_replies_in_order():
    c = FakeLLM(["a", "b"])
    assert c.chat([{"role": "user", "content": "1"}]) == "a"
    assert c.chat([{"role": "user", "content": "2"}]) == "b"
    assert len(c.calls) == 2

def test_siliconflow_missing_key_raises_clearly(monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.setattr("ragv1.embedding.ENV_FILE_CANDIDATES", ())
    with pytest.raises(LLMNotConfigured):
        SiliconFlowClient()

def test_siliconflow_does_not_silently_noop(monkeypatch):
    """⚠️ Review Focus 4 的根：生成层缺席绝不能退化成 no-op。"""
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.setattr("ragv1.embedding.ENV_FILE_CANDIDATES", ())
    with pytest.raises(RuntimeError):
        SiliconFlowClient()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_qa_llm.py -v`
Expected: FAIL — 模块不存在

- [ ] **Step 3: 在 `ragv1/config.py` 加 QA 配置**

```python
# ── V2 QA 层 ──────────────────────────────────────────────
QA_LLM_MODEL = os.environ.get("QA_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
QA_LLM_TIMEOUT = float(os.environ.get("QA_LLM_TIMEOUT", "60"))
QA_TOP_N = int(os.environ.get("QA_TOP_N", "8"))              # 送进 prompt 的块数
REFUSAL_SCORE_THRESHOLD = float(os.environ.get("REFUSAL_SCORE_THRESHOLD", "0.0"))  # dev 集上调
QA_VERIFY_SUPPORT = os.environ.get("QA_VERIFY_SUPPORT", "0") not in {"0", "false", "no"}
```

- [ ] **Step 4: 实现 `ragv1/qa/llm.py`**

`SiliconFlowClient.__init__` 调 `resolve_api_key()`，把 `RuntimeError` 转成
`LLMNotConfigured`（`LLMNotConfigured` 继承 `RuntimeError`，所以两条测试都过）。
`chat` 用 `openai.OpenAI(base_url=BASE_URL).chat.completions.create(...)` 并返回 `choices[0].message.content`。
`FakeLLM` 按序弹出 `replies`，用尽时抛 `AssertionError("LLM 调用次数超出预期")`——
这样"空 history 不该调 LLM"这类断言能真的失败。

- [ ] **Step 5: 跑测试确认通过**

Run: `py -m pytest tests/test_qa_llm.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add ragv1/qa/__init__.py ragv1/qa/llm.py ragv1/config.py tests/test_qa_llm.py
git commit -m "feat(qa): LLM 客户端（缺 key 明确报错，不静默降级）"
```

---

### Task 10: 提示词与响应解析

**Files:**
- Create: `ragv1/qa/prompt.py`
- Test: `tests/test_qa_prompt.py`

**Interfaces:**
- Consumes: `types.Chunk`（仅用 `text` / `chunk_id` / `doc_id` / `page`）
- Produces:
  - `prompt.render_context(chunks: list[Chunk]) -> str`（`[1] 正文...`）
  - `prompt.build_messages(question: str, context: str, history: tuple[Turn, ...]) -> list[dict]`
  - `prompt.parse_answer(raw: str) -> Answer`（失败抛 `prompt.AnswerParseError`）
  - `types.Answer(text: str, citation_ids: tuple[int, ...], refused: bool)`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qa_prompt.py
import pytest
from ragv1.qa import prompt
from ragv1.types import Chunk, Turn

def _c(cid, text):
    return Chunk(chunk_id=cid, doc_id="d", heading_path=(), text=text)

def test_render_context_numbers_from_one():
    ctx = prompt.render_context([_c("a", "甲"), _c("b", "乙")])
    assert ctx.startswith("[1] 甲") and "[2] 乙" in ctx

def test_build_messages_includes_history_before_question():
    msgs = prompt.build_messages("追问", "[1] 甲", (Turn("user", "上文"), Turn("assistant", "答复")))
    joined = " ".join(m["content"] for m in msgs)
    assert joined.index("上文") < joined.index("追问")

def test_parse_answer_reads_plain_json():
    a = prompt.parse_answer('{"answer": "是 42", "citations": [1], "insufficient": false}')
    assert a == prompt.Answer(text="是 42", citation_ids=(1,), refused=False)

def test_parse_answer_strips_markdown_fence():
    a = prompt.parse_answer('```json\n{"answer":"x","citations":[],"insufficient":true}\n```')
    assert a.refused is True

def test_parse_answer_raises_on_garbage():
    with pytest.raises(prompt.AnswerParseError):
        prompt.parse_answer("我不知道你在说什么")

def test_parse_answer_rejects_out_of_range_citation():
    with pytest.raises(prompt.AnswerParseError, match="越界"):
        prompt.parse_answer('{"answer":"x","citations":[9],"insufficient":false}')
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_qa_prompt.py -v`
Expected: FAIL — 模块 / `Answer` 不存在

- [ ] **Step 3: 在 `ragv1/types.py` 加 `Answer`，实现 `prompt.py`**

`Answer` 定义在 `types.py`，`prompt.py` 再导出一次（测试里用 `prompt.Answer`）。

系统提示词三条硬约束，原文照写：
**只依据下方上下文回答** / **每个论断附 `[n]` 标记** / **上下文不足时把 `insufficient` 置 true，不要推测**。
响应格式要求 JSON：`{"answer": str, "citations": [int], "insufficient": bool}`。

`parse_answer`：先剥 ```` ```json ```` 围栏 → `json.loads` → 校验
`citations` 每一项是 int 且在 `[1, len(context)]` 内（**上下文长度从 raw 里拿不到，
所以由调用方在 `generate` 里再校验一次**——本函数只校验"是正整数"，
越界校验见 Step 4 的实现说明）→ 返回 `Answer`。
解析任何一步失败 → `AnswerParseError`。

> **越界校验的归属**：`parse_answer` 需要 `len(context)` 才能判越界。
> 因此签名改为 `parse_answer(raw: str, n_context: int) -> Answer`，
> `test_parse_answer_rejects_out_of_range_citation` 传 `n_context=2` 且 citations 为 `[9]`。

- [ ] **Step 4: 跑测试确认通过**

Run: `py -m pytest tests/test_qa_prompt.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ragv1/qa/prompt.py ragv1/types.py tests/test_qa_prompt.py
git commit -m "feat(qa): 编号上下文提示词与结构化响应解析"
```

---

### Task 11: 生成（含两层拒答）

**Files:**
- Create: `ragv1/qa/generate.py`
- Test: `tests/test_qa_generate.py`

**Interfaces:**
- Consumes: `qa.llm.LLMClient`、`qa.prompt`、`types.FusedHit` / `Chunk` / `Answer`
- Produces: `generate.answer_question(
    question: str, hits: list[FusedHit], chunks: list[Chunk],
    client: LLMClient, history: tuple[Turn, ...] = (),
    top_n: int = QA_TOP_N, refusal_threshold: float = REFUSAL_SCORE_THRESHOLD,
) -> tuple[Answer, dict[str, str]]`
（第二个返回值是降级码字典，键为 `"qa"`，沿用项目"降级不静默"的写法）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qa_generate.py
from ragv1.qa.generate import answer_question
from ragv1.qa.llm import FakeLLM
from ragv1.types import Chunk, FusedHit

def _c(cid, text="正文"):
    return Chunk(chunk_id=cid, doc_id="d", heading_path=(), text=text)

def _fh(cid, rank=1, score=0.03):
    return FusedHit(chunk_id=cid, rrf_score=score, sources=("vector",), rank=rank)

OK = '{"answer":"是 42 [1]","citations":[1],"insufficient":false}'

def test_returns_answer_with_citations():
    a, deg = answer_question("问", [_fh("a")], [_c("a")], FakeLLM([OK]))
    assert a.text == "是 42 [1]" and a.citation_ids == (1,) and deg == {}

def test_empty_context_refuses_without_calling_llm():
    """⚠️ Review Focus 2：上下文为空必须拒答，且不该浪费一次 LLM 调用。"""
    llm = FakeLLM([])                       # 任何调用都会因越界抛 AssertionError
    a, deg = answer_question("问", [], [], llm)
    assert a.refused is True and a.citation_ids == ()

def test_low_fused_score_refuses_even_with_context():
    """检索侧门控：融合分低于阈值 → 拒答。"""
    a, _ = answer_question("问", [_fh("a", score=0.001)], [_c("a")], FakeLLM([]),
                           refusal_threshold=0.05)
    assert a.refused is True

def test_model_insufficient_flag_alone_causes_refusal():
    """生成侧门控：上下文不为空、分数够，但模型说不够 → 也算拒答。"""
    raw = '{"answer":"不知道","citations":[],"insufficient":true}'
    a, _ = answer_question("问", [_fh("a")], [_c("a")], FakeLLM([raw]))
    assert a.refused is True

def test_parse_failure_degrades_instead_of_raising():
    a, deg = answer_question("问", [_fh("a")], [_c("a")], FakeLLM(["不是 JSON"]))
    assert a.refused is True and deg["qa"].startswith("qa_parse_failed")

def test_llm_exception_propagates():
    """⚠️ 与 rewrite 的降级行为形成对照：答案缺席不能静默。"""
    import pytest
    class Boom:
        calls = []
        def chat(self, messages, **kw): raise RuntimeError("网络断了")
    with pytest.raises(RuntimeError):
        answer_question("问", [_fh("a")], [_c("a")], Boom())

def test_context_is_truncated_to_top_n():
    hits = [_fh(f"c{i}", rank=i + 1) for i in range(12)]
    chunks = [_c(f"c{i}") for i in range(12)]
    llm = FakeLLM([OK])
    answer_question("问", hits, chunks, llm, top_n=3)
    ctx = [m for m in llm.calls[0] if "上下文" in m["content"] or "[1]" in m["content"]]
    assert "[4]" not in " ".join(m["content"] for m in llm.calls[0])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_qa_generate.py -v`
Expected: FAIL — 模块不存在

- [ ] **Step 3: 实现 `answer_question`**

顺序：
1. `chunks` 为空 → 直接返回 `Answer("", (), refused=True)`，**不调 LLM**。
2. 取 `hits[:top_n]`，按 `chunk_id` 对齐 `chunks`。
3. `max(h.rrf_score) < refusal_threshold` → 拒答，**不调 LLM**。
4. 否则 `render_context` → `build_messages` → `client.chat` →
   `parse_answer(raw, n_context=len(used))`。
   解析失败 → `Answer("", (), refused=True)` + `{"qa": "qa_parse_failed: ..."}`。
5. `answer.refused` 或 `insufficient` → `refused=True`。
6. `client.chat` 抛出的异常**向上传播**，不捕获。

- [ ] **Step 4: 跑测试确认通过**

Run: `py -m pytest tests/test_qa_generate.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add ragv1/qa/generate.py tests/test_qa_generate.py
git commit -m "feat(qa): 带引用的生成与两层拒答"
```

---

### Task 12: 引用校验

**Files:**
- Create: `ragv1/qa/verify.py`
- Modify: `ragv1/types.py`（`VerifyReport`）
- Test: `tests/test_qa_verify.py`

**Interfaces:**
- Consumes: `types.Answer` / `VerifyReport`
- Produces:
  - `verify.verify_citations(answer: Answer, n_context: int) -> VerifyReport`（机械，零成本）
  - `verify.verify_support(answer, chunks, client) -> tuple[str, ...]`（LLM，可选）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qa_verify.py
from ragv1.qa.verify import verify_citations
from ragv1.qa.llm import FakeLLM
from ragv1.types import Answer, Chunk, VerifyReport

def test_ok_when_every_marker_has_a_citation_and_vice_versa():
    r = verify_citations(Answer("是 42 [1]", (1,), False), n_context=2)
    assert r.ok is True and r.invalid_markers == ()

# ⚠️ Review Focus 3：正文写了 [3]，citations 里却没有 3
def test_marker_in_text_missing_from_citations_is_invalid():
    r = verify_citations(Answer("见 [1] 和 [3]", (1,), False), n_context=3)
    assert r.ok is False and 3 in r.invalid_markers

def test_citation_not_mentioned_in_text_is_invalid():
    r = verify_citations(Answer("见 [1]", (1, 2), False), n_context=3)
    assert r.ok is False and 2 in r.invalid_markers

def test_out_of_range_citation_is_invalid():
    r = verify_citations(Answer("见 [5]", (5,), False), n_context=3)
    assert r.ok is False and 5 in r.invalid_markers

def test_refused_answer_is_ok_regardless():
    r = verify_citations(Answer("", (), True), n_context=0)
    assert r.ok is True

def test_repeated_marker_counts_once():
    r = verify_citations(Answer("见 [1] 又见 [1]", (1,), False), n_context=1)
    assert r.ok is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_qa_verify.py -v`
Expected: FAIL — 模块 / `VerifyReport` 不存在

- [ ] **Step 3: 在 `ragv1/types.py` 加 `VerifyReport`，实现 `verify.py`**

```python
@dataclass(frozen=True)
class VerifyReport:
    ok: bool
    invalid_markers: tuple[int, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
```

`verify_citations`：用 `re.findall(r"\[(\d+)\]", answer.text)` 取正文标记集合，
与 `set(answer.citation_ids)` 取对称差；再并入越界项；排序去重后放进 `invalid_markers`，
`ok = not invalid_markers`。`refused=True` → 直接 `VerifyReport(ok=True)`。

`verify_support`：把答案按句切分，连同引用块一起交给 LLM 判定，
返回不被支持的句子元组。**默认不在主链路上调用**（成本翻倍）。

- [ ] **Step 4: 跑测试确认通过**

Run: `py -m pytest tests/test_qa_verify.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ragv1/qa/verify.py ragv1/types.py tests/test_qa_verify.py
git commit -m "feat(qa): 引用一致性机械校验"
```

---

### Task 13: 生成与校验接入图

**Files:**
- Modify: `ragv1/orchestration/supervisor.py`
- Test: `tests/test_supervisor_qa.py`

**Interfaces:**
- Consumes: `qa.generate.answer_question`、`qa.verify.verify_citations`、`types.Answer` / `VerifyReport` / `Turn`
- Produces:
  - `build_graph(retriever, paths, tracer=None, text_lookup=None, llm=None)`
    —— `llm is None` 时**图里没有** generate / verify 节点
  - `State` 新增 `history: tuple[Turn, ...]`、`answer: Answer | None`、`verified: VerifyReport | None`
  - `run_ask_meta(graph, query, k, history=(), on_error=ON_ERROR_SKIP) -> tuple[Answer | None, dict[str, str]]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_supervisor_qa.py
import pytest
from ragv1.orchestration.supervisor import build_graph, run_query_meta, run_ask_meta
from ragv1.qa.llm import FakeLLM

OK = '{"answer":"是 42 [1]","citations":[1],"insufficient":false}'

def test_llm_none_means_no_qa_nodes_in_graph(retriever):
    """节点存在与否由构造期决定 —— 与 paths 同一套规矩。"""
    g = build_graph(retriever, {"vector"})
    assert "generate" not in g.get_graph().nodes

def test_llm_given_adds_qa_nodes(retriever, text_lookup):
    g = build_graph(retriever, {"vector"}, text_lookup=text_lookup, llm=FakeLLM([OK]))
    assert {"generate", "verify"} <= set(g.get_graph().nodes)

def test_search_path_still_works_unchanged(retriever):
    fused, deg = run_query_meta(build_graph(retriever, {"vector"}), "q", 5)
    assert deg == {}

def test_run_ask_meta_returns_answer_and_degraded(retriever, text_lookup):
    g = build_graph(retriever, {"vector"}, text_lookup=text_lookup, llm=FakeLLM([OK]))
    answer, deg = run_ask_meta(g, "q", 5)
    assert answer is not None and deg == {}

def test_run_ask_meta_with_empty_index_refuses(retriever, text_lookup):
    g = build_graph(retriever, {"graph"}, text_lookup=text_lookup, llm=FakeLLM([]))
    answer, _ = run_ask_meta(g, "查不到的东西", 5)
    assert answer.refused is True
```

（`retriever` / `text_lookup` 两个 fixture 用现有的内存假 store 构造，复用 `tests/test_supervisor.py` 里的做法。）

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_supervisor_qa.py -v`
Expected: FAIL — `build_graph() got an unexpected keyword argument 'text_lookup'`

- [ ] **Step 3: 实现**

- `build_graph` 新增 `text_lookup` / `llm` 两个可选参数；
  `llm is not None` 时 `add_node("generate", ...)`、`add_node("verify", ...)`，
  并把边改成 `FUSE → generate → verify → END`（`llm is None` 时保持 `FUSE → END`）。
- `_generate_node`：用 `text_lookup` 取块原文（取不到 → 该块跳过），调 `answer_question`。
- `_verify_node`：调 `verify_citations`；异常 → 降级 `{"verified": None, "degraded": {"qa": "verify_failed: ..."}}`。
- `run_ask_meta`：`graph.invoke({... "history": history, "k": k, ...})`，
  返回 `(state.get("answer"), dict(state.get("degraded") or {}))`。

- [ ] **Step 4: 跑测试确认通过并全量回归**

Run: `py -m pytest -q`
Expected: 全部 passed

- [ ] **Step 5: Commit**

```bash
git add ragv1/orchestration/supervisor.py tests/test_supervisor_qa.py
git commit -m "feat(orchestration): generate/verify 节点按构造期参数入图"
```

---

### Task 14: `/ask` 端点

**Files:**
- Modify: `ragv1/api/main.py`
- Modify: `ragv1/api/server.py`
- Test: `tests/test_ask_api.py`

**Interfaces:**
- Consumes: `build_graph` / `run_ask_meta`、`llm.SiliconFlowClient`
- Produces:
  - `create_app(retriever, meta_lookup=None, text_lookup=None, llm=None)`
  - `POST /ask` → `AskResponse`
  - `server.build_app(idx, embed_fn=None, llm=None)`
  - `AskResponse.llm_configured: bool`（False 时 `/ask` 返回 503）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_ask_api.py
from fastapi.testclient import TestClient
from ragv1.api.main import create_app
from ragv1.qa.llm import FakeLLM

OK = '{"answer":"是 42 [1]","citations":[1],"insufficient":false}'

def test_ask_returns_answer_and_citations(retriever, meta_lookup, text_lookup):
    app = create_app(retriever, meta_lookup=meta_lookup, text_lookup=text_lookup,
                     llm=FakeLLM([OK]))
    r = TestClient(app).post("/ask", json={"query": "q", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] and body["citations"][0]["marker"] == 1
    assert body["refused"] is False

# ⚠️ Review Focus 4：LLM 未配置必须明确报错，绝不能静默降级成只有检索结果的响应
def test_ask_without_llm_returns_503_not_a_degraded_200(retriever, meta_lookup, text_lookup):
    app = create_app(retriever, meta_lookup=meta_lookup, text_lookup=text_lookup, llm=None)
    r = TestClient(app).post("/ask", json={"query": "q"})
    assert r.status_code == 503
    assert "LLM" in r.json()["detail"]

def test_search_endpoint_unchanged_when_llm_present(retriever, meta_lookup, text_lookup):
    app = create_app(retriever, meta_lookup=meta_lookup, text_lookup=text_lookup,
                     llm=FakeLLM([OK]))
    assert TestClient(app).post("/search", json={"query": "q"}).status_code == 200

def test_ask_rejects_blank_query(retriever, meta_lookup, text_lookup):
    app = create_app(retriever, meta_lookup=meta_lookup, text_lookup=text_lookup, llm=FakeLLM([OK]))
    assert TestClient(app).post("/ask", json={"query": "   "}).status_code == 422

def test_ask_reports_rewritten_query_as_null_without_history(retriever, meta_lookup, text_lookup):
    app = create_app(retriever, meta_lookup=meta_lookup, text_lookup=text_lookup, llm=FakeLLM([OK]))
    assert TestClient(app).post("/ask", json={"query": "q"}).json()["rewritten_query"] is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_ask_api.py -v`
Expected: FAIL — `create_app() got an unexpected keyword argument 'text_lookup'`

- [ ] **Step 3: 实现**

- `create_app` 新增 `text_lookup` / `llm` 参数；图缓存键改为 `(frozenset(paths), llm is not None)`。
- `/ask` 请求体 `AskRequest(query, k, history, paths)`；
  `llm is None` → `HTTPException(503, "LLM 未配置：/ask 需要可用的生成模型")`；
  空 query → 422（与 `/search` 一致）。
- `AskResponse(answer, citations, refused, rewritten_query, verified, degraded)`；
  `citations` 由 `answer.citation_ids` 映射回 `chunk_id` + `meta_lookup` 得到的元数据。
- `server.build_app` 传 `text_lookup=fts.text_of`；`llm` 缺省时
  **惰性构造** `SiliconFlowClient()`，构造失败不拦启动，只是 `/ask` 会 503。

- [ ] **Step 4: 跑测试确认通过并全量回归**

Run: `py -m pytest -q`
Expected: 全部 passed

- [ ] **Step 5: Commit**

```bash
git add ragv1/api/main.py ragv1/api/server.py tests/test_ask_api.py
git commit -m "feat(api): /ask 端点（LLM 未配置时明确 503）"
```

---

### Task 15: 答案级指标

**Files:**
- Create: `ragv1/evaluation/answer_metrics.py`
- Test: `tests/test_answer_metrics.py`

**Interfaces:**
- Consumes: `types.Answer` / `VerifyReport` / `qaset.EvalItem`
- Produces:
  - `answer_metrics.citation_accuracy(answers: list[tuple[Answer, VerifyReport]]) -> float`
  - `answer_metrics.refusal_accuracy(results: list[tuple[str, bool, bool]]) -> dict[str, float]`
    输入为 `(category, expected_answerable, refused)`；
    返回 `{"refusal_accuracy", "over_refusal_rate"}`
  - `answer_metrics.summarize(refused_flags: list[bool]) -> float`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_answer_metrics.py
from ragv1.evaluation.answer_metrics import citation_accuracy, refusal_accuracy
from ragv1.types import Answer, VerifyReport

def test_citation_accuracy_is_ratio_of_ok_reports():
    pairs = [(Answer("a [1]", (1,), False), VerifyReport(ok=True)),
             (Answer("a [2]", (2,), False), VerifyReport(ok=False, invalid_markers=(2,)))]
    assert citation_accuracy(pairs) == 0.5

def test_citation_accuracy_of_empty_input_is_zero_not_crash():
    assert citation_accuracy([]) == 0.0

def test_refusal_accuracy_counts_only_unanswerable():
    rows = [("unanswerable", False, True),    # 正确拒答
            ("unanswerable", False, False),   # 漏答
            ("fact", True, False)]            # 可回答，正确作答
    m = refusal_accuracy(rows)
    assert m["refusal_accuracy"] == 0.5

def test_over_refusal_rate_counts_only_answerable():
    rows = [("fact", True, True),             # 误拒
            ("fact", True, False),
            ("unanswerable", False, True)]
    m = refusal_accuracy(rows)
    assert m["over_refusal_rate"] == 0.5

def test_both_rates_are_zero_when_the_relevant_class_is_absent():
    m = refusal_accuracy([("fact", True, False)])
    assert m["refusal_accuracy"] == 0.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_answer_metrics.py -v`
Expected: FAIL — 模块不存在

- [ ] **Step 3: 实现**

两个函数都是纯统计，无 IO。分母为 0 时返回 `0.0`（**不是** `nan`——
`nan` 会让报告静默失真）。`expected_answerable` 由 `category != "unanswerable"` 推出，
但**显式作为参数传入**，避免指标逻辑依赖分类命名。

- [ ] **Step 4: 跑测试确认通过**

Run: `py -m pytest tests/test_answer_metrics.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add ragv1/evaluation/answer_metrics.py tests/test_answer_metrics.py
git commit -m "feat(evaluation): 引用准确率与拒答率指标"
```

---

# 里程碑 M4 · 多轮

### Task 16: 查询改写

**Files:**
- Create: `ragv1/qa/rewrite.py`
- Modify: `ragv1/config.py`
- Test: `tests/test_qa_rewrite.py`

**Interfaces:**
- Consumes: `qa.llm.LLMClient`、`types.Turn`、`config.QA_HISTORY_MAX_TURNS`
- Produces: `rewrite.rewrite(query: str, history: tuple[Turn, ...], client: LLMClient) -> str | None`
  （返回 `None` 表示"无需改写"，调用方据此短路）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qa_rewrite.py
import pytest
from ragv1.qa.rewrite import rewrite
from ragv1.qa.llm import FakeLLM
from ragv1.types import Turn

def test_empty_history_returns_none_without_calling_llm():
    """空 history 必须短路 —— FakeLLM([]) 一被调用就会抛。"""
    assert rewrite("它怎么配？", (), FakeLLM([])) is None

def test_nonempty_history_returns_rewritten_query():
    llm = FakeLLM(["XX 模块怎么配？"])
    assert rewrite("它怎么配？", (Turn("user", "讲讲 XX 模块"),), llm) == "XX 模块怎么配？"

def test_whitespace_only_reply_is_treated_as_no_rewrite():
    assert rewrite("它怎么配？", (Turn("user", "上文"),), FakeLLM(["   "])) is None

def test_same_as_original_is_treated_as_no_rewrite():
    assert rewrite("它怎么配？", (Turn("user", "上文"),), FakeLLM(["它怎么配？"])) is None

def test_history_is_truncated_to_max_turns():
    llm = FakeLLM(["新的"])
    turns = tuple(Turn("user", f"第{i}轮") for i in range(20))
    rewrite("问", turns, llm)
    sent = " ".join(m["content"] for m in llm.calls[0])
    assert "第0轮" not in sent          # 最早的轮次必须被截掉
    assert "第19轮" in sent

def test_llm_exception_propagates_to_caller():
    """改写失败由**调用方**（图节点）降级，本函数不吞异常。"""
    class Boom:
        calls = []
        def chat(self, m, **k): raise RuntimeError("超时")
    with pytest.raises(RuntimeError):
        rewrite("问", (Turn("user", "上文"),), Boom())
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_qa_rewrite.py -v`
Expected: FAIL — 模块不存在

- [ ] **Step 3: 在 `ragv1/config.py` 加 `QA_HISTORY_MAX_TURNS = int(os.environ.get("QA_HISTORY_MAX_TURNS", "6"))`，实现 `rewrite`**

`history` 为空 → 返回 `None`（**在构造 prompt 之前就返回**）。
否则取最后 `QA_HISTORY_MAX_TURNS` 轮构造 prompt，要求模型"只输出一个自包含的查询，
不要解释"；回复去空白后若为空、或与原查询相同 → 返回 `None`。

- [ ] **Step 4: 跑测试确认通过**

Run: `py -m pytest tests/test_qa_rewrite.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ragv1/qa/rewrite.py ragv1/config.py tests/test_qa_rewrite.py
git commit -m "feat(qa): 多轮查询改写（空历史短路）"
```

---

### Task 17: 双路检索接入图

**Files:**
- Modify: `ragv1/orchestration/supervisor.py`
- Test: `tests/test_two_variant_retrieval.py`

**Interfaces:**
- Consumes: `qa.rewrite.rewrite`、`fusion.router` 的 `#rw` 折扣
- Produces:
  - `State.query_variants: list[str]`（初始 `[query]`；改写后 `[query, rewritten]`）
  - `ranked` 键规则：变体 0 用裸路径名，变体 1 用 `f"{path}#rw"`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_two_variant_retrieval.py
from ragv1.orchestration.supervisor import _retrieval_node, State

class CountingRetriever:
    def __init__(self):
        self.vector = self
        self.queries = []
    def retrieve(self, query, k):
        self.queries.append(query)
        return []

def test_single_variant_uses_bare_path_key():
    r = CountingRetriever()
    out = _retrieval_node(r, "vector")({"query": "q", "query_variants": ["q"], "k": 3})
    assert set(out["ranked"]) == {"vector"}

def test_two_variants_use_bare_and_rw_keys():
    r = CountingRetriever()
    out = _retrieval_node(r, "vector")({"query": "q", "query_variants": ["q", "q 改写"], "k": 3})
    assert set(out["ranked"]) == {"vector", "vector#rw"}
    assert r.queries == ["q", "q 改写"]

# ⚠️ Review Focus 5：第二变体失败不能拖垮整次查询，也不能污染第一变体的降级标记
def test_second_variant_failure_degrades_only_that_variant():
    class HalfBoom(CountingRetriever):
        def retrieve(self, query, k):
            if query == "炸":
                raise RuntimeError("embedding 挂了")
            return super().retrieve(query, k)
    r = HalfBoom()
    out = _retrieval_node(r, "vector")({"query": "q", "query_variants": ["q", "炸"], "k": 3})
    assert "vector" in out["ranked"] and "vector#rw" not in out["ranked"]
    assert "vector" not in out["degraded"]
    assert out["degraded"]["vector#rw"].startswith("RuntimeError")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_two_variant_retrieval.py -v`
Expected: FAIL — `_retrieval_node` 不读 `query_variants`

- [ ] **Step 3: 改 `_retrieval_node`，并加 `rewrite` 节点**

- `_retrieval_node` 遍历 `state["query_variants"]`：索引 0 → 键 `path`，
  索引 ≥1 → 键 `f"{path}{REWRITTEN_SUFFIX}"`。**每个变体单独 try**，
  失败只给该键写空列表 + 该键的降级码。
- 新增 `_rewrite_node`：`history` 为空 → 原样返回 `{"query_variants": [query]}`（不调 LLM）；
  否则调 `rewrite`，异常 → `{"query_variants": [query], "degraded": {"rewrite": "rewrite_failed: ..."}}`；
  返回 `None` → `[query]`；否则 `[query, rewritten]`。
- 边：`START → rewrite → 各检索路`（`llm` 为 None 时无 rewrite 节点，`START → 各检索路` 不变）。

- [ ] **Step 4: 跑测试确认通过并全量回归**

Run: `py -m pytest -q`
Expected: 全部 passed

- [ ] **Step 5: Commit**

```bash
git add ragv1/orchestration/supervisor.py tests/test_two_variant_retrieval.py
git commit -m "feat(orchestration): 改写变体双路检索（单变体失败不影响主路）"
```

---

# 里程碑 M5 · 验收

### Task 18: 验收脚本

**Files:**
- Create: `scripts/eval_v2.py`
- Test: `tests/test_eval_v2.py`

**Interfaces:**
- Consumes: `api.server.build_retriever` / `FtsStore`、`qa.generate` / `qa.verify`、`evaluation.answer_metrics`、`evaluation.harness`
- Produces:
  - `eval_v2.retrieval_table(retriever, items, weights) -> str`（Markdown 表）
  - `eval_v2.answer_table(rows) -> str`
  - `--retrieval-only` / `--set PATH` / `--index-dir PATH` / `--with-support` 四个开关

- [ ] **Step 1: 写失败测试**

```python
# tests/test_eval_v2.py
import importlib.util
from pathlib import Path

def _load():
    spec = importlib.util.spec_from_file_location(
        "eval_v2", Path(__file__).resolve().parents[1] / "scripts" / "eval_v2.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_retrieval_table_lists_every_path_row():
    mod = _load()
    from ragv1.types import FusedHit
    rows = {"vector": [FusedHit("a", 0.1, ("vector",), 1, {})],
            "fused": [FusedHit("a", 0.2, ("vector",), 1, {})]}
    table = mod.retrieval_table(["a"], rows)
    assert "| vector |" in table and "| fused |" in table

def test_answer_table_reports_rates_with_two_decimals():
    mod = _load()
    t = mod.answer_table({"citation_accuracy": 1.0, "refusal_accuracy": 0.666})
    assert "1.00" in t and "0.67" in t
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -m pytest tests/test_eval_v2.py -v`
Expected: FAIL — 文件不存在

- [ ] **Step 3: 实现 `scripts/eval_v2.py`**

复用 `evaluation/harness.py` 的 `_recall` / `_mrr`（**不要重写指标**——
两处实现漂移会让新旧数字不可比）。`--retrieval-only` 只跑检索表；
否则跑完整 `/ask` 链路并输出答案表。**必须打印 LLM 调用次数**（成本可见）。

- [ ] **Step 4: 跑测试确认通过**

Run: `py -m pytest tests/test_eval_v2.py -v`
Expected: 2 passed

- [ ] **Step 5: 跑完整验收（人工步骤）**

Run: `py scripts/eval_v2.py --index-dir .indexes/kb_zh --set ragv1/evaluation/qa_sets/zh_test.jsonl --with-support`
Expected: 产出判据 1–4 所需的全部数字。

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_v2.py tests/test_eval_v2.py
git commit -m "feat(evaluation): V2 验收脚本"
```

---

### Task 19: 验收报告与文档

**Files:**
- Create: `ACCEPTANCE_V2.md`
- Modify: `README.md`
- Test: 无（文档任务）

**Interfaces:**
- Consumes: Task 18 的全部实测数字
- Produces: 判据 1–5 的逐条判定

- [ ] **Step 1: 写 `ACCEPTANCE_V2.md`**

逐条列判据 1–5 与实测数字，**沿用 V1 的标准：不通过的判据照实写，并写明边界**。
必须包含：
- 权重搜索范围与最终 `STATIC_FUSION_WEIGHTS`
- `REFUSAL_SCORE_THRESHOLD` 在 **dev** 集上的调参过程（交代没碰 test 集）
- 引用支持度抽检的**样本量**
- 一次 `/ask` 的 LLM 调用次数与 token
- 已知限制（新评估集规模、LLM-judge 的偏差、单次运行不下结论）

- [ ] **Step 2: 更新 `README.md`**

取舍表新增行：**为什么从等权 RRF 改加权**、**为什么改写成双路而不是替换**、
**为什么生成节点失败要抛出而 rewrite 失败降级**。
「已知限制」里把 V1 的"判据 1 未通过"更新为 V2 的实测结果。

- [ ] **Step 3: 最终全量回归**

Run: `py -m pytest -q`
Expected: 全部 passed（含 V1 的 231 条）

Run: `py -m pytest -m smoke -v -s`
Expected: 真实引擎的实测输出照常打印

- [ ] **Step 4: Commit**

```bash
git add ACCEPTANCE_V2.md README.md
git commit -m "docs: V2 验收报告与 README 更新"
```

---

## Self-Review

**1. Spec coverage**

| Spec 章节 | 覆盖任务 |
|---|---|
| §4.1 加权融合 + contributions | Task 5 |
| §4.2 查询路由（含 bm25 方向、改写折扣） | Task 6, 7 |
| §4.3.1 LLM 客户端（缺 key 报错、可注入） | Task 9 |
| §4.3.2 prompt + 结构化解析 + `qa_parse_failed` | Task 10 |
| §4.3.3 generate + 两层拒答 | Task 11 |
| §4.3.4 verify（机械 + 可选支持度） | Task 12 |
| §4.4 多轮改写 | Task 16 |
| §4.5 图与 State、节点构造期决定、降级边界、键命名 | Task 13, 17 |
| §4.6 /ask 与 503、text_lookup、图缓存键 | Task 14 |
| §5.1 中文语料 + 人工集（含不可回答/多跳/dev-test 分离） | Task 1, 2, 3, 4 |
| §5.2 四个指标 | Task 15 |
| §6 判据 1–5 | Task 8（判据1）、Task 18、19 |
| §7 测试策略 | 各任务的 Step 1 |
| §9 里程碑 | 五个里程碑 |

无遗漏。

**2. Step scan** — 每个 Step 都是单一动作：写测试 / 跑测试看失败 / 实现 / 跑测试看通过 / commit。实现步骤给的是签名与"测试没确定的那部分"，没有替实现者写函数体。

**3. Type consistency** — `Turn` 在 Task 2 定义、Task 10/11/13/16/17 复用，签名一致。`Answer` 在 Task 10 定义（`text/citation_ids/refused`），Task 11/12/14/15 引用一致。`VerifyReport` 在 Task 12 定义（`ok/invalid_markers/unsupported_claims`），Task 15 引用一致。`route_weights(query, ranked, static)` 在 Task 6 定义、Task 7 调用，参数顺序一致。`parse_answer(raw, n_context)` 在 Task 10 的 Step 3 内修正，Step 1 的测试已按此写。

**4. Review Focus** — 五条全部落成测试：① → Task 6 的 `test_dominant_fulltext_hit_upweights_fulltext_bm25_is_negative`；② → Task 11 的 `test_empty_context_refuses_without_calling_llm` + `test_low_fused_score_refuses_even_with_context` + `test_model_insufficient_flag_alone_causes_refusal`；③ → Task 12 的 `test_marker_in_text_missing_from_citations_is_invalid` + `test_citation_not_mentioned_in_text_is_invalid`；④ → Task 14 的 `test_ask_without_llm_returns_503_not_a_degraded_200`；⑤ → Task 17 的 `test_second_variant_failure_degrades_only_that_variant`。

**5. Proportion** — 计划约为 spec 的 2.5 倍，绝大部分是测试断言（spec 的 §7 把测试策略交给计划写）。实现步骤只给签名，没有函数体抄写。

## 已知的未决项

- **LLM API key 是否可用尚未确认**（spec §8 的头号风险）。Task 9 起需要真 key 才能产出真实数字；在此之前所有 QA 任务靠 `FakeLLM` 离线跑，测试仍须全绿。
- **`STATIC_FUSION_WEIGHTS` 的初值是别家语料上的数字**，Task 8 必须在本项目 dev 集上重调，**最终值以实测为准，不得沿用调研值**。
