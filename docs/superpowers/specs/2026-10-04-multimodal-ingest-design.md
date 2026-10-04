# 多模态入库：图片与表格的正确解析

日期：2026-10-04
分支：`feat/multimodal-ingest`
状态：待评审

## 1. 背景与现状（实测）

当前入库链路只有一条，入口只认 `.md`：

```
rglob("*.md") → parse_document(text) → ParsedDoc(blocks) → chunk_document(1200) → list[Chunk] → 三路 store
```

对开发语料（`ragflow-main/docs`，107 篇）实测：

| 项 | 数字 | 现状后果 |
|---|---|---|
| 表格 | 152 张（23 篇）；**9 张超 1200 字符（5.9%）**；最长 3582 字符 | 与正文同属一个 section，检索命中的是「标题+正文+表格」混合块；超限时被按**字符**硬切，行被拦腰截断且后续块**丢表头** |
| 图片 | 184 处引用（51 篇），全部是**远程 URL** | 只有 `alt` 文字进了 chunk，视觉内容全丢；本机无像素，想 OCR 也得先下载 |
| 非 md 文件 | 28 json + 9 mdx | 被 `rglob("*.md")` 直接跳过 |
| PDF | 0 | **完全没有入口** |

关键代码位置：

- [build.py:39](../../../ragv1/ingest/build.py#L39) `rglob("*.md")` —— 唯一的入口过滤
- [chunker.py:57-69](../../../ragv1/ingest/chunker.py#L57-L69) `_paragraphs` 按**空行**分段 → 表格行之间无空行 → 整张表是「一个段落」→ 超 budget 时 `para[i:i+budget]` 硬切
- [types.py:27-38](../../../ragv1/types.py#L27-L38) `Chunk` 只有 4 个字段，无任何来源元数据
- [fts_store.py:36-41](../../../ragv1/store/fts_store.py#L36-L41) `chunks` 表（回源表）只有 4 列；[vector_store.py:38-42](../../../ragv1/store/vector_store.py#L38-L42) 不存 metadata；`Hit`/`FusedHit` 无 metadata 字段

**降级机制的先例已经存在**：三路检索已有 `degraded` 语义（[test_degradation.py](../../../tests/test_degradation.py)），本设计在入库侧沿用同一套词汇。

### 环境可行性（已核查）

PDF 解析所需依赖**本机全部已装**，无需新增重量级依赖：

| 包 | 状态 | 用途 |
|---|---|---|
| pdfplumber 0.11.10 | ✅ | `chars`(带 bbox) / `images` / `find_tables()` |
| pypdfium2 | ✅ | 页面渲染成位图 |
| onnxruntime 1.29.0 | ✅ | RapidOCR 运行时 |
| Pillow 12.3.0 / pypdf 6.19.0 | ✅ | 图像处理 / PDF 元信息 |

**唯一需新装**：`rapidocr-onnxruntime`（纯 pip）。`reportlab` 仅在**生成 PDF 测试 fixture 时**用一次，不进运行时依赖。

## 2. 目标与非目标

### 目标

1. 图片：版面分析区分「有文字层」/「纯图片」；有文字层且质量好的直接用，不重复解析
2. 图片双通道：OCR（提取文字）+ 多模态（类型/内容/数据/关系），按内容分流，兼有则都产出，最后合并并恢复阅读顺序
3. 表格：从正文拆出、结构化（Markdown 表）、独立成 chunk、**不被定长切片截断**，大表按「表头+若干行」切且每块带表头
4. 复杂表格解析不出时降级为「保留原文 + 未结构化标记」，不丢弃
5. 新增 PDF 入口（pdfplumber 版面分析 + pypdfium2 渲染扫描页）
6. 每个 chunk 带来源元数据（文档名/页码/位置），图片另带原图引用与本地缓存路径；**端到端暴露到 API**
7. 任何解析失败都落降级标记，不静默丢弃
8. 引擎（OCR / 多模态）可配置、可注入，默认选本机能跑通的
9. 模块拆分：版面分析 / OCR / 多模态 / 表格解析 各自独立可替换
10. 可运行测试：带文字的图、流程图、含表格的文档三份 fixture

### 非目标（本期不做）

- 引入 ML 版面模型（PP-StructureV3 / DocLayout-YOLO）—— 依赖重、Windows 装 paddle 不稳，与「默认能跑通」冲突。接口留 `LayoutEngine` 便于以后替换
- 图片入库进向量库做多模态检索（用图搜图）—— 本期只把图片**转为文本**参与文本检索
- OCR/多模态结果的置信度校正、人工复核流程
- 重排（cross-encoder）—— 与本期无关

## 3. 设计总览

```
文件 (.md / .pdf)
   │
   ├─ loader.py        按扩展名分派
   │    ├─ MarkdownLoader ─┐
   │    └─ PdfLoader ──────┤
   │                       ↓
   ├─ layout.py        版面分析 → 有序 Element 流
   │                   （文字层判定：有文字层且质量好 → 直接用，跳过双通道）
   │                       ↓
   ├─ table.py       ┌─ 表格 → Markdown 表 → 表头重复切分
   ├─ image_source.py├─ 图片取回（URL 下载 / 本地路径）+ 缓存
   ├─ ocr.py         ├─ OCR 通道
   ├─ vlm.py         └─ 多模态通道
   │                       ↓
   │                  按 order 合并，恢复阅读顺序
   │                       ↓
   └─ elements.py      Element 流 → list[Chunk]（表格块绕开字符硬切）
                           ↓
                  三路 store（FTS 加列 / Chroma metadatas）+ 元数据端到端暴露
```

## 4. 数据模型

### 4.1 Element（新增，`ragv1/types.py`）

```python
@dataclass(frozen=True)
class Element:
    """版面分析产出的有序元素。三种 kind 的载荷放在同一张扁平表里，
    用 None 区分——比再套一层继承好读，也与 Block/Chunk 的风格一致。"""

    kind: str                       # text | table | image
    order: int                      # 全文档阅读顺序，0 起
    text: str                       # text=正文 / table=Markdown 表 / image=最终描述文本
    heading_path: tuple[str, ...] = ()
    page: int | None = None         # PDF 从 1 计；Markdown 为 None
    bbox: tuple[float, float, float, float] | None = None   # (x0, top, x1, bottom)
    # ── image 专有 ──
    image_ref: str | None = None    # 原图 URL / 相对路径
    image_path: str | None = None   # 本地缓存路径
    # ── table 专有 ──
    table_structured: bool = True   # False = 降级为保留原文
    # ── 降级 ──
    degrade: tuple[str, ...] = ()   # 原因码，已排序去重
```

### 4.2 Chunk（扩展，字段全部带默认值）

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    heading_path: tuple[str, ...]
    text: str
    # ── 新增：来源元数据 ──
    kind: str = "text"                                        # text | table | image
    page: int | None = None                                   # PDF 页码；Markdown 为 None
    order: int = 0                                            # 全文档阅读序号
    part: int = 0                                             # 同一元素被切成多块时的序号（0 起）
    bbox: tuple[float, float, float, float] | None = None      # 位置（Markdown 为 None）
    image_ref: str | None = None                              # 原图 URL / 相对路径
    image_path: str | None = None                             # 本地缓存路径
    table_structured: bool = True                             # False = 未结构化降级
    degrade: tuple[str, ...] = ()                             # 降级原因码
```

**全字段带默认值**是硬要求：现有测试与代码里 `Chunk(chunk_id=..., doc_id=..., heading_path=..., text=...)` 的构造点一律不用改。

## 5. 模块清单（`ragv1/ingest/`）

| 文件 | 职责 | 对外接口 |
|---|---|---|
| `loader.py` | 按扩展名分派；`MarkdownLoader` / `PdfLoader` | `load(path, cfg) -> list[Element]`；`dispatch(path) -> Loader \| None` |
| `layout.py` | 版面分析：PDF 页 → 有序区域 + 文字层判定 | `analyze_page(page) -> list[Element]`；`has_text_layer(page) -> bool` |
| `ocr.py` | OCR 引擎抽象 + RapidOCR 默认实现 | `OcrEngine.extract(bytes) -> OcrResult` |
| `vlm.py` | 多模态引擎抽象 + SiliconFlow 默认实现 | `VisionEngine.describe(bytes) -> VisionResult` |
| `table.py` | 表格 → Markdown；表头重复切分；降级 | `parse_table(rows) -> str`；`split_table(md, budget) -> list[str]` |
| `image_source.py` | 图片取回（URL/本地）+ 缓存 | `fetch(ref, base_dir, cache_dir, fetcher) -> bytes \| None` |
| `elements.py` | Element 流 → `list[Chunk]`（三路共用） | `chunk_elements(elements, doc_id, max_chars) -> list[Chunk]` |
| `degrade.py` | 降级原因码常量表 | 常量 + `ALL_CODES` |

**保留不动的**：`parser.py`（Markdown 文本 → Block）继续被 `MarkdownLoader` 复用；`chunker.py` 的 `make_chunk_id` / `_iter_sections` / `_group_text` 保留，纯文本路径继续走它。

### 5.1 引擎接口与配置

```python
class OcrEngine(Protocol):
    def extract(self, image: bytes) -> OcrResult: ...   # OcrResult(text, confidence, lines)

class VisionEngine(Protocol):
    def describe(self, image: bytes) -> VisionResult: ...  # VisionResult(image_type, content, data, relations, raw)
```

`config.py` 新增（全部可从环境变量覆盖）：

```python
OCR_ENGINE   = "rapidocr"       # rapidocr | none
VLM_ENGINE   = "siliconflow"    # siliconflow | none
VLM_MODEL    = "Qwen/Qwen2.5-VL-32B-Instruct"
OCR_MIN_CHARS = 10              # 判「文字为主」的最小字符数
OCR_MIN_CONF  = 0.5             # 判「文字层质量好」的最小置信度
ENABLE_IMAGE  = True
IMAGE_CACHE_DIR = INDEX_DIR / "images"   # 进 .gitignore
```

`rapidocr` 走**惰性 import**（首次调用才导入）；未配置 `SILICONFLOW_API_KEY` 时 `VLM_ENGINE` 自动降为 `none` 并落 `vlm_unavailable` 标记。

### 5.2 多模态结构化 Prompt

要求 VLM **只输出 JSON**：

```json
{"image_type": "流程图|图表|截图|照片|其他",
 "content": "图片的核心内容 1-3 句",
 "data": "图中关键数据/数值/结论，无则空串",
 "relations": "元素关系（流程走向/层级/对比），无则空串"}
```

返回非法 JSON → 落 `vlm_bad_json`，**原始返回文本当作 content 保留**（不丢）。

## 6. 关键规则

### 6.1 Markdown 侧的识别规则

`MarkdownLoader` 逐行扫描（**复用 `parser._strip_frontmatter` 与现有围栏代码块跟踪**），维护标题栈得到 `heading_path`，把每一行归入三类：

| 行形态 | 前提 | 产出 |
|---|---|---|
| `#`~`######` 标题行 | 不在代码围栏内 | 更新标题栈，不产出 Element |
| 连续 **≥2 行**以 `\|` 开头（允许前导空白），且第 2 行匹配分隔行 `^\s*\|[\s\-:|]+\|\s*$` | 不在代码围栏内 | 一个 `table` Element |
| **整行**（strip 后）恰为 `![alt](ref)` | 不在代码围栏内 | 一个 `image` Element |
| 其余 | — | 累积进当前 text 段 |

- **代码围栏内的 `\|` 行与 `![...]` 行一律不识别**——技术文档里代码块遍地，误判会污染正文
- **行内图片不开 Element**：`前面文字 ![alt](ref) 后面文字` 保持原样留在正文（语料中罕见，拆开反而丢上下文）
- **引用式图片** `![alt][id]`（需查文末定义表）本期不解析：整行保留在正文 + `degrade(image_ref_unresolved)`
- 图片 `ref` 是 `http(s)://` → 走下载；否则按**该 md 文件所在目录**解析为本地路径
- 表格与图片是**文本连续段唯一的断开点**（见 §8.1）

### 6.2 图片分流（需求 1.1 / 1.3）

先判**文字层**（PDF 场景）：图片 bbox 落在页文字层字符上且有文本 → 直接产出 text Element，**两个通道都不跑**。

进入通道的图片，按下列显式规则表分流（避免玄学阈值）：

| OCR 有料¹ | VLM 可用 | VLM 判定 | 输出 |
|---|---|---|---|
| 是 | 否 | — | OCR 文本 + `degrade(vlm_unavailable)` |
| 是 | 是 | 纯文字截图 | 仅 OCR 文本 |
| 是 | 是 | 图表/流程图/照片 | **OCR 文本 + 多模态描述（都产出）** |
| 否 | 是 | — | 仅多模态描述 |
| 否 | 否 | — | 占位文本 + `degrade(ocr_empty, vlm_unavailable)` |

¹ 「OCR 有料」= `len(text.strip()) >= OCR_MIN_CHARS` 且 `confidence >= OCR_MIN_CONF`。

**合并格式**（进 chunk 的 text，只含非空段）：

```
[图片] {alt 或 image_ref}
类型: {image_type}
内容: {content}
数据: {data}
关系: {relations}
文字: {ocr_text}
```

**恢复阅读顺序**：所有 Element 按 `order` 排序合并；通道可并行，排序在最后做。

### 6.3 表格切分（需求 2）

```
输入：Markdown 表 = 表头行 + 分隔行 + N 个数据行
budget = MAX_CHARS - len(heading 前缀)

整表 ≤ budget        → 一块（part=0）
否则                 → 每块 = 表头行 + 分隔行 + 贪心装入的数据行
                       每块都重复表头，part 递增
单行本身 > budget    → 该行单独成块，不切，落 degrade(table_row_too_long)
                        （宁可超限也不切断行——这是"不被拦腰截断"的兑现）
非合法 pipe 表       → table_structured=False，整块原文保留
                       + degrade(table_unstructured)
```

### 6.4 降级原因码（`degrade.py`）

解析类：`table_unstructured` `table_row_too_long` `image_ref_unresolved`
取图类：`image_fetch_failed` `image_missing`
OCR 类：`ocr_failed` `ocr_empty` `ocr_low_conf`
多模态类：`vlm_unavailable` `vlm_failed` `vlm_bad_json`
PDF 类：`pdf_page_no_text_layer` `pdf_render_failed` `pdf_table_bbox_missing`

**每个 chunk 的 `degrade` 有序去重**，保证输出可复现；测试断言 `set(degrade) ⊆ ALL_CODES`。

### 6.5 版面分析（PDF，自研轻量版）

用 pdfplumber 的 `page.chars`（每字符带 bbox）/ `page.images` / `page.find_tables()` 做**区域聚类**：

1. chars 按 y 聚类成行，行按间距聚成段
2. `find_tables()` 的 bbox → table 区域；`images` 的 bbox → image 区域
3. 段与区域求交判定归属
4. **无任何 chars 的页**（扫描件）→ pypdfium2 渲染整页 → 整页走图片双通道，落 `pdf_page_no_text_layer`
5. 渲染失败 → `pdf_render_failed`，该页产出占位 Element（不静默丢）

**已知局限（如实记录）**：不是 ML 版面模型，**双栏/多栏排版**的聚类可能切歪；依赖依赖注入的 `LayoutEngine` 接口，后续可替换。

## 7. 存储与元数据端到端

### 7.1 FtsStore

- `chunks` 表新增列：`kind, page, order, part, bbox, image_ref, image_path, table_structured, degrade`
- `degrade` / `bbox` 以 JSON 字符串存储
- **迁移策略**：新增 `schema_version` 表；版本不符则 `DROP TABLE` 重建。索引是**派生的、永远全量重建**的数据，静默丢弃无损失（此理由写进注释）
- 新增 `meta_of(chunk_id) -> dict | None`（回源表的自然延伸）

### 7.2 VectorStore

`add()` 传 `metadatas=`（Chroma metadata 值只允许 str/int/float/bool → `degrade` 用 JSON 字符串）。

### 7.3 端到端暴露（API）

**设计取舍**：`rrf_fuse` 只做排名、不认识 store（[rrf.py](../../../ragv1/fusion/rrf.py) 的文档契约）。因此**不把 store 塞进融合层**，而是在 API 边界注入查表函数：

```python
def create_app(retriever, meta_lookup: Callable[[str], dict] | None = None): ...
```

- `meta_lookup(chunk_id)` 的返回键**固定为**：`doc_id, kind, page, order, part, bbox, image_ref, image_path, table_structured, degrade`（其中 `bbox` / `degrade` 已反序列化；chunk 不存在时返回 `None`）
- `meta_lookup` 缺省 `None` → 行为与现在完全一致（现有 [test_api.py](../../../tests/test_api.py) 不受影响）
- `ResultItem` 新增：`doc_id, kind, page, order, bbox, image_ref, image_path, table_structured, degrade`
- 生产接线由 `build_corpus` 的同级装配代码传入 `FtsStore.meta_of`

### 7.4 GraphStore

不动（只消费 `heading_path`）。

## 8. 兼容性与索引重建

### 8.1 纯文本不变性（硬约束，可测）

对**不含表格/图片**的文档，新链路产出必须与旧链路**逐字节一致**（同样的 `chunk_id`、同样的 `text`）。

保证手段两条：

1. **文本连续段合并**：同一 `heading_path` 下**连续的 text Element 合并成一个正文段**，再走原有的 `_group_text`。表格/图片是唯一的「断开点」——所以纯文本文档的正文段与旧 `_iter_sections` 的 body 完全相同
2. `make_chunk_id(doc_id, heading_path, index)` **签名不变**，`index` 仍是全局递增序号

回归测试：对一组不走网络的内联 markdown 样本，断言新旧产出的 `(chunk_id, text)` 序列 diff 为空。

### 8.2 必须全量重建索引

**含图片/表格的文档（51 篇含图 + 23 篇含表），块数量变化导致其后所有块的 `index` 移位 → `chunk_id` 改变。**

- 这不是缺陷，是「给新内容腾出块位」的必然结果
- 代价：旧 `.indexes/` 作废，必须 `build_corpus` 全量重建
- FTS 的 `schema_version` 守卫会把这一点变成显式行为，而不是静默错读

### 8.3 破坏现有路径的检查清单

- [ ] 现有 107 个测试全绿
- [ ] 纯文本等价性回归测试通过
- [ ] `Chunk` 新增字段全部有默认值，无构造点被破坏
- [ ] `create_app` 的 `meta_lookup` 缺省时 API 行为不变

## 9. 测试计划

### 9.1 Fixtures（`tests/fixtures/`，随仓库提交）

| 文件 | 内容 | 生成方式 |
|---|---|---|
| `ocr_sample.png` | 带文字的截图（白底黑字，含数字与中文） | Pillow 生成一次后提交 |
| `flow.png` | 流程图（方框 + 箭头 + 标签） | Pillow 画框/箭头生成后提交 |
| `table_doc.md` | 含小表 + 超 1200 字符大表 + 正常正文 | 手写 |
| `table_doc.pdf` | 带文字层与表格的小 PDF | reportlab **仅在生成 fixture 时**使用，产物提交 |

**Hermetic 原则**：单测一律注入 **fake OCR / fake VLM / fake fetcher**，不碰网络、不花钱、结果确定。真实引擎的验证另做（见 9.3）。

### 9.2 单元测试

| 文件 | 覆盖 |
|---|---|
| `tests/test_table.py` | 表格识别 / Markdown 化 / 表头重复切分 / **单行超长不切** / 非 pipe 表降级 |
| `tests/test_image_channels.py` | 6.2 分流表**逐行覆盖**（5 种组合）；合并格式；`vlm_bad_json` 时原文保留 |
| `tests/test_image_source.py` | URL 下载并缓存 / 缓存命中不重复下载 / 本地相对路径 / 取图失败落码 |
| `tests/test_elements.py` | Element→Chunk；**纯文本等价性回归**；表格块绕开字符硬切 |
| `tests/test_layout.py` | 文字层判定（有/无）；区域分类；扫描页渲染降级（fake 页对象） |
| `tests/test_loader.py` | 扩展名分派；未知扩展名跳过；`.md`/`.pdf` 分别路由 |
| `tests/test_meta.py` | 元数据落 FTS/Chroma；`meta_of` 回源；API 带出（`meta_lookup` 注入） |
| `tests/test_degrade_codes.py` | 所有产出码 ⊆ `ALL_CODES`；有序去重 |

### 9.3 真实引擎 smoke（opt-in）

标记 `@pytest.mark.smoke`，**默认跳过**（`pytest -m smoke` 显式运行）：

- RapidOCR 真跑 `ocr_sample.png`，断言抽出的文字包含预期关键字
- SiliconFlow VLM 真跑 `flow.png`（需 `SILICONFLOW_API_KEY`），断言返回合法 JSON 且 `image_type` 合理

**这两条的产出（实际文字、实际描述）会写进验收报告**，而不是只说「跑通了」。

## 10. 分期

| 阶段 | 内容 | 新依赖 | 可验收产物 |
|---|---|---|---|
| **一** | 元数据骨架（Chunk 扩字段 → 三 store 落地 → API 暴露）+ 表格解析（Markdown 路径） | **无** | 表格独立成块、大表带表头、纯文本等价性回归绿 |
| **二** | 图片双通道（OCR + VLM）+ 图片取回缓存 + PDF Loader（版面分析 + 渲染） | `rapidocr-onnxruntime` | 三份 fixture 全通 + smoke 实测数字 |

两阶段各自可独立验收；阶段一不引入任何新依赖，立刻可测。

## 11. 风险与未决

| 风险 | 应对 |
|---|---|
| `rapidocr-onnxruntime` 首次运行需下载模型（联网） | 惰性 import + 失败落 `ocr_failed`；smoke 单独跑 |
| 远程图片下载慢/失败（184 处） | 本地缓存 + 失败落码不中断入库；测试用注入 fetcher |
| 自研版面分析在多栏 PDF 上切歪 | 接口 `LayoutEngine` 可替换；局限写进 README 已知限制 |
| VLM 调用花钱 | fake 引擎用于测试；真实调用只在 smoke；结果可缓存 |
| 语料目录硬编码在 [config.py](../../../ragv1/config.py) | 沿用现状，本期不改（已在 README 已知限制中） |

## 12. 验收标准

1. 三份 fixture 各有可运行的测试，`pytest` 全绿（现有 107 + 新增）
2. **含表格文档**：表格独立成 chunk；超 1200 字符的大表被切成多块且**每块都带表头**；不存在被切断的行
3. **带文字的图**：OCR 文本进入 chunk 且可被检索命中
4. **流程图**：多模态描述含类型/内容/关系四要素（smoke 实测）
5. 每个新 chunk 带 `doc_id` + `page`/`order`（+ PDF 的 `bbox`）；图片 chunk 带 `image_ref` 与 `image_path`
6. API 返回体带出来源元数据（`meta_lookup` 注入时）
7. 所有降级路径都有对应测试，且无静默丢弃
8. 纯文本产出与改造前逐字节一致（回归测试为证）
