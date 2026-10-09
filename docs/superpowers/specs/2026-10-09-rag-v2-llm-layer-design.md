# rag-v2 设计：给三层检索补上 LLM 层

> 日期：2026-10-09 · 分支：`feat/v2-llm-layer` · 状态：待评审

## 1. 背景

rag-v1 是一个**检索内核**：三路并行召回（向量 / 全文 / 图谱）+ RRF 融合，
链路里 **0 次 LLM 调用**。这个定位是刻意的，但它带来两个问题：

1. **读起来像"只做了一半"**。"RAG 项目"在面试官与简历筛选的默认预期里，
   指的是一个**能回答问题的系统**。一个只有 `/search` 的服务需要额外解释才
   能对上预期，而筛选阶段没有解释的机会。
2. **V1 有一个自己承认没通过的判据**：融合在 A/C/D 三类上**回退**于最强单路
   （`ACCEPTANCE.md` §2，A 类 −0.200）。根因已定位为 RRF 等权稀释强路。

V1 的诚实是它最大的资产，但"诚实"和"完整"是两回事。V2 补上后者。

### 1.1 调研结论（决定 V2 形状的三条）

外部调研（GitHub 高星项目 + 2025–2026 技术演进 + 国内 JD 关键词）给出三条结论：

- **Tier 1 基本要求**（不是加分项）：混合检索 + 加权融合 + 重排 + **引用溯源**。
  V1 有前两项（但融合在稀释），后两项空白。
- **等权 RRF 打不过最强单路是业界反复复现的已知现象**，不是本项目的实现 bug。
  成熟做法是加权 RRF / 查询自适应权重——Milvus 官方文档给出的企业三信号示例
  是 `w_vector=0.50, w_bm25=0.35, w_kg=0.15`，与本项目三层结构一一对应。
- **V1 的 A/B/C/D 评估集是结构性自证**：问题由 chunk 标题套模板生成
  （`qa_gen.py:50-77`），关键词强重叠、几乎必中，且全是单跳事实问题。
  **在修好评估集之前，任何"LLM 层 work 了"的结论都不可信。**

> ⚠️ 上述外部结论中，论文编号与具体数字来自网络检索，**方向可靠但未在本机复现**。
> 写进 README / 简历 / 面试引用前必须先在本项目语料上跑出实测数字。

## 2. 目标与非目标

### 2.1 目标

把 rag-v1 从一个检索内核，补成一个**可验证的** RAG 问答系统。

### 2.2 范围

**做：**

- 加权融合（含查询自适应权重）—— 直接修 V1 未通过的判据
- 生成层：基于检索结果的回答，带 **chunk 级引用标记**
- **拒答**：知识库没有时明确说不知道
- **多轮对话**：指代消解 / 查询改写
- **评估集重建**：dev/test 分离、真实问法、多跳题、不可回答问题、难负例
- 新增中文主语料（Dify / FastGPT 中文 docs）
- 新增答案级指标：faithfulness / citation_accuracy / refusal_accuracy

**不做（YAGNI，明确出局）：**

- Agentic 自省循环（自评-重检索）
- 权限过滤 / 多租户
- 增量索引
- store 层全局锁（44 QPS 瓶颈）的修复
- 图片进向量库做以图搜图
- 更换向量库 / 引入外部图数据库
- 修改 `/search` 的既有行为

## 3. 架构总览

```
                         ┌─→ vector ─┐
query ──[rewrite]───────→┼─→ fulltext┼─→ [加权 fuse] ─→ [generate] ─→ [verify] ─→ answer
          ↑              └─→ graph ──┘        ↑              ↑
      对话历史                            route_weights   text_lookup
     （无历史则短路）                      + 加权 RRF      + LLM
```

**关键结构决定：节点存在与否由构造期决定，不靠节点内部 `if`。**

这条沿用 V1 已有的哲学——`build_graph` 只给**启用的**路径建节点
（`supervisor.py:82-83`，"未启用的通路根本不在图里，从结构上保证它不会被调用"）。
V2 把同一条规矩用在 LLM 层：

- `llm=None` → 图里**没有** rewrite / generate / verify 节点 → 这就是 V1 的检索图
- `llm=...` → 图里**有**这三个节点 → 这是问答图

因此 `/search` 与 `/ask` 走的是**同一个 `build_graph`**，只是参数不同。
不存在"两套编排"的问题。

## 4. 组件设计

### 4.1 加权融合 · `fusion/rrf.py`

**接口变化：**

```python
def rrf_fuse(
    ranked: dict[str, list[Hit]],
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> list[FusedHit]
```

**契约（必须逐条测）：**

- `weights=None` 或 `weights={}` → **与 V1 行为逐字节一致**。这是零回归的前提。
- 出现在 `ranked` 里但**不在** `weights` 里的路径 → 权重按 **1.0**（默认）计。
- 权重 ≤ 0 的路径 → 视为不参与融合（显式关闭，而不是"贡献负分"）。
- `weights` 里的未知路径名 → 静默忽略（不报错；路径集合由调用方决定）。
- 平局判据仍是 `chunk_id` 升序——**可复现性这条不能破**。

**公式：** `score(chunk) = Σ_path  w[path] / (k + rank_in_path)`

**`FusedHit` 新增字段：**

```python
contributions: dict[str, float] = field(default_factory=dict)
```

记录每一路对这个块的贡献分（`w[path] / (k + rank)`）。
**为什么值得加**：这是回答"这个块为什么排第一"的唯一依据，也是调权重时
唯一能看出"强路被谁挤掉"的地方。默认空字典，构造方式向后兼容。

### 4.2 查询路由 · `fusion/router.py`（新）

```python
def route_weights(
    query: str,
    ranked: dict[str, list[Hit]],
    static: dict[str, float] | None = None,
) -> dict[str, float]
```

**纯规则、零 LLM、确定性**（与项目既有的实体消解同一条取舍：技术场景下
规则可复现且更准）。

**信号（从便宜到贵，依次施加）：**

1. **静态基线**：取自 `config.STATIC_FUSION_WEIGHTS`，由 dev 集网格搜索得到。
   起点采用调研查到的企业三信号示例 `vector=0.50, fulltext=0.35, graph=0.15`。
2. **词面命中信号**：query 与某块标题近似一致、或含标识符特征
   （下划线 / 驼峰 / 全大写缩写 / 反引号包裹）→ 上调全文路、下调向量路。
3. **全文路置信度**：全文路 top-1 的 bm25 分数显著高于该路历史中位数 →
   同样上调全文路。

> ⚠️ **bm25 在本项目里是负值**（越小越相关，`rrf.py` 的注释已写明"FTS5 bm25 负值"）。
> 路由判据必须先统一方向，否则会把"最相关"读成"最不相关"。这一点单测钉死。

**改写变体的折扣**：多轮改写产生的第二变体（键名 `path#rw`）权重 = 同路径权重
× `config.REWRITE_WEIGHT_DISCOUNT`（初值 0.5）。取舍理由见 §4.4。

**为什么路由发生在 fuse 之前而不是更早**：信号 2、3 需要看到检索结果，
所以它是**检索后、融合前**的一步。这正好落在 `_fuse_node` 里。

### 4.3 生成层 · `qa/`（新包）

```
ragv1/qa/
├── llm.py        LLM 客户端（硅基流动，OpenAI 兼容）
├── prompt.py     带编号上下文的模板 + 拒答指令
├── generate.py   生成带引用的答案
└── verify.py     引用校验
```

#### 4.3.1 `llm.py`

```python
class LLMClient(Protocol):
    def chat(self, messages: list[dict], **kwargs) -> str: ...
```

- 生产实现走硅基流动的 OpenAI 兼容端点，配置项（base_url / model / 超时）全部
  从环境变量读，默认值放 `config.py`。模型名初值取 `Qwen/Qwen2.5-72B-Instruct`
  （与 `config.py` 既有的 Qwen 命名风格一致），可用环境变量替换；
  具体哪个模型可跑通以你账号下的实际可用列表为准。
- **无 API key 时抛明确异常，不静默降级。**
  取舍理由：Langfuse 无 key 时 no-op 是对的（埋点缺席不影响结果正确性）；但
  生成层缺席会产出一个**看起来正常却是错的**答案。两者后果不同，处理方式就不能相同。
- 客户端**可注入**——与 `embed_fn` 同一约定，否则整条 QA 链路无法离线测试。

#### 4.3.2 `prompt.py`

- 把 top-N 融合块渲染成编号上下文：`[1] <文本>\n\n[2] <文本>...`
- 系统提示词的三条硬约束：**只依据上下文回答** · **每个论断附 `[n]` 标记** ·
  **上下文不足时必须拒答而不是推测**
- 拒答用**结构化输出**而非魔法字符串：要求模型返回
  ```json
  {"answer": "...", "citations": [1, 3], "insufficient": false}
  ```
  解析失败 → 落降级码 `qa_parse_failed`，**不抛异常**（沿用项目"解析失败落降级码、
  不静默丢弃"的一贯做法）。

#### 4.3.3 `generate.py`

```python
@dataclass(frozen=True)
class Answer:
    text: str
    citation_ids: tuple[int, ...]   # 1-based，指向上下文编号
    refused: bool
```

**引用溯源采"前置约束生成"**：编号上下文直接进 prompt，让模型输出时就带标记，
而不是先生成答案再回头做归因（后者更贵、更不可控、且归因本身又是一次 LLM 猜测）。

**拒绝两层兜，两层都触发才算拒答：**

| 层 | 判据 | 位置 |
|---|---|---|
| 检索侧 | 融合分低于阈值 → 上下文整体不可信 | `generate` 入口，零成本 |
| 生成侧 | 模型返回 `insufficient: true` | 解析结果 |

只做一层会误拒：仅看融合分，会让"分数低但其实答得出"的问题被拒；
仅看模型，会让模型在空上下文上依然编出答案。

**阈值取值**：初值放 `config.REFUSAL_SCORE_THRESHOLD`，**在 dev 集上调**，
不在 test 集上调（否则又是一次自欺）。调参过程与最终取值都要记进验收报告。

#### 4.3.4 `verify.py`

**两层，成本不同，分开开关：**

1. **机械校验（零成本，永远开）**：每个 `citation_id` 必须落在
   `1..len(context)` 内，且答案正文里确实出现对应 `[n]` 标记。
   越界/对不上 → 记为 invalid，计入 `citation_accuracy`。
2. **支持度判定（LLM，默认关）**：判断每个论断是否真被引用块支持。
   默认关闭的理由是**成本**——它会让一次 `/ask` 的 LLM 调用翻倍。
   评估脚本里显式打开。

### 4.4 多轮 · `qa/rewrite.py`

```python
def rewrite(query: str, history: list[Turn]) -> str | None
```

- **`history` 为空 → 返回 `None`，短路，零 LLM 调用。** 单轮查询的延迟特性
  与 V1 完全一致。
- 有历史 → LLM 输出一个去指代的独立查询（"它怎么配？" → "XX 模块怎么配？"）。

**双路检索（关键取舍）**：改写查询**不替换**原始查询，而是**两个都检索**，
结果一起进融合。理由：调研里点名 HyDE 一类改写对模糊问题会"幻觉出一个假文档
把检索带偏"——改写有同样的风险。双路是标准缓解，且**加权 RRF 天然支持它**：
把 `vector` 与 `vector#rw` 当成两个独立排序列表即可，无需额外机制。

这也是 V1 那条经验的延续：**朴素替换/组合会稀释已经 work 的那一路**
（`RERANK_AB.md` 末尾的总结）。改写变体降权而非等权，是同一个教训的应用。

### 4.5 图与 State 变化 · `orchestration/supervisor.py`

```python
class State(TypedDict):
    query: str
    query_variants: list[str]        # 新增：[原始] 或 [原始, 改写]
    history: list[Turn]              # 新增
    k: int
    on_error: str
    ranked: Annotated[dict[str, list[Hit]], _merge_dict]
    degraded: Annotated[dict[str, str], _merge_dict]
    weights: dict[str, float]        # 新增：route_weights 的输出，便于观测
    fused: list[FusedHit]
    answer: Answer | None            # 新增
    verified: VerifyReport | None    # 新增
```

**`query_variants` 由谁写：**

- 初始 state 由调用方（`run_ask`）填成 `[query]` —— 单轮时的唯一变体
- `rewrite` 节点在有历史时覆写为 `[query, rewritten]`
- 每个检索节点**遍历 `query_variants`**，对每个变体各调一次本路检索，
  把结果按下面的键名写进 `ranked`

**`ranked` 的键命名（向后兼容规则）：**

- 变体 0（原始查询）→ 键就是**裸路径名** `vector` / `fulltext` / `graph`
- 变体 1（改写查询）→ 键是 `vector#rw` 等

这样**单轮查询的 `ranked` 与 V1 完全一致**，既有测试不需要改。
这条兼容规则要有专门的测试钉住。

**两个新类型：**

```python
@dataclass(frozen=True)
class Turn:
    role: str      # "user" | "assistant"
    text: str

@dataclass(frozen=True)
class VerifyReport:
    ok: bool                        # 机械校验是否全部通过
    invalid_markers: tuple[int, ...]  # 越界或在正文中找不到的标记
    unsupported_claims: tuple[str, ...]  # 支持度判定为不支持的论断（判定关闭时为空）
```

**图拓扑：**

```
START ─┬─(history 非空)→ rewrite ─┬→ vector ──┐
       └─(history 为空)──────────┼→ fulltext ┼→ fuse → generate → verify → END
                                 └→ graph ───┘
```

**降级的边界（必须与非降级严格区分）：**

| 节点 | 失败时 | 理由 |
|---|---|---|
| `rewrite` | 降级：退回只用原始查询，记 `rewrite_failed` | 改写是增强，缺席不影响可用性 |
| 三路检索 | 降级：该路缺席（**V1 既有行为，不动**） | 同上 |
| `generate` | **抛出** | 没有答案就是没有答案，返回空答案等于隐瞒 |
| `verify` | 降级：返回答案但标记 `verified: false` | 校验是附加信息，缺席不该阻断答案 |

### 4.6 API · `api/main.py`

**`/search` 一字不改**（向后兼容）。

**新增 `/ask`：**

```python
class AskRequest(BaseModel):
    query: str
    k: int = Field(default=DEFAULT_K, ge=1, le=100)
    history: list[Turn] | None = None
    paths: list[str] | None = None

class Citation(BaseModel):
    marker: int            # 对应答案里的 [n]
    chunk_id: str
    # 其余来源元数据与 ResultItem 同源（doc_id / page / order / image_ref ...）

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool
    rewritten_query: str | None    # 如实告知改写成了什么
    verified: bool
    degraded: dict[str, str]
```

- `text_lookup`（取块原文喂给生成层）**注入到 `build_graph`**，来源是
  `FtsStore.text_of`——与既有的 `meta_lookup` 同一模式（`api/main.py:10-12`）。
  这样融合层与生成层都不认识 store，"store 依赖止步于 API 边界"这条不破。
- 空的 `query` 仍返回 422（与 `/search` 一致）。
- 图缓存键由 `frozenset(paths)` 扩为 `(frozenset(paths), qa_enabled)`。
- 新增 `run_ask_meta(graph, query, k, history, ...)`，与既有 `run_query_meta`
  （`supervisor.py:105`）**同形**：返回 `(answer, degraded)`，把降级情况如实交回
  调用方。`/search` 继续用 `run_query_meta`，两者互不影响。

## 5. 评估重建

**这是 V2 最花时间的一节，也是能不能"证明系统 work"的唯一依据。**

### 5.1 两套评估集

| 评估集 | 语料 | 用途 | 构成 |
|---|---|---|---|
| **结构性集** | 英文（RAGFlow docs） | **回归基准**，保证新旧可比 | 现有 A/B/C/D，**不动** |
| **人工集** | 中文（Dify/FastGPT docs） | **主评估** | 见下 |

**人工集构成（目标 30–50 条，dev/test 分离）：**

- 真实用户问法（**不用标题套模板**）
- 多跳题、比较题、时序题
- **不可回答问题**（答案不在库里）——测拒答
- **难负例**（字面相似但不对的块）
- 部分条目带 `history`——测多轮

**标注格式：** `{question, answer_chunk_ids, category, history?}`，
落在 `evaluation/qa_sets/`，dev/test 分开两个文件。

> ⚠️ 工作量要如实预估：**标注是人力活，不是脚本活**。要先建好中文索引，
> 再人工读语料写问题并标答案块。这是 V2 的关键路径，不是可以压缩的尾巴。

### 5.2 新增指标 · `evaluation/answer_metrics.py`

| 指标 | 定义 | 测法 |
|---|---|---|
| `faithfulness` | 答案里被引用块支持的论断占比 | LLM 判定（评估脚本内开） |
| `citation_accuracy` | `citation_id` 合法且与正文 `[n]` 对得上的比例 | 机械校验，**可完全自动** |
| `refusal_accuracy` | 不可回答问题上的正确拒答率 | 机械判定 |
| `over_refusal_rate` | 可回答问题上被误拒的比例 | 机械判定 |

**`citation_accuracy` 与 `refusal_*` 是机械可测的**——这是 V2 最硬的部分：
不依赖任何 LLM 裁判，因此不可争辩。

## 6. 验收判据

写进 `ACCEPTANCE.md`，逐条给实测数字（沿用 V1 的诚实标准：**不通过的判据照实写**）。

1. **加权融合 ≥ 最强单路**，在**新的人工集**上成立
   —— 这是 V1 明确未通过的判据，V2 要在新集上翻盘。同时在结构性集上不回归。
2. **引用可验证**：`citation_accuracy == 1.0`（机械可测）；
   支持度人工抽检并给出样本量。
3. **拒答正确**：不可回答问题拒答率过线，**同时**可回答问题上误拒率不过线
   （只报前者是作弊）。
4. **多轮有效**：改写前后检索指标有可复现的提升（同评估集、同种子）。
5. **零回归**：231 个 V1 测试全绿；英文语料上 A/B/C/D 各指标不下降。

## 7. 测试策略

沿用 V1 的 TDD 与"先看着它失败"。

**必须有的边界测试：**

- `rrf_fuse(weights=None)` 输出与 V1 **逐字节一致**（回归护栏）
- 权重 ≤ 0 的路径被排除；未知路径名被忽略
- **bm25 负值方向**：路由不会把"最相关"读成"最不相关"
- `ranked` 键命名兼容：单轮时是裸路径名，多轮时是 `path#rw`
- `rewrite` 在空 history 下**不调用 LLM**（用计数假客户端钉住）
- `generate` 的 JSON 解析失败 → 落 `qa_parse_failed` 降级码，不抛
- `generate` 的 LLM 异常 → **抛出**（与 rewrite 的降级行为形成对照）
- 空上下文 → 必须拒答，绝不返回编造的答案
- `citation_id` 越界 → 计入 invalid，不崩溃
- LLM 客户端可注入 → 全程离线可测（无 key 时测试仍全绿）

## 8. 风险与依赖

| 风险 | 影响 | 应对 |
|---|---|---|
| **LLM API key 未确认** | 阻塞生成层 | **动手前先确认**。无 key 时整条 QA 链路只能靠假客户端测，无法产出真实数字 |
| 人工标注工作量被低估 | 关键路径延期 | 先小批（10 条）跑通全链路，再扩到 30–50 条 |
| 新评估集又写得太容易 | 重蹈结构性自证 | 强制包含不可回答问题、难负例、多跳题；dev/test 分离 |
| LLM 调用成本 | 评估跑不动 | 默认关闭支持度判定；评估脚本加调用计数与预算上限 |
| 中文语料可获取性 | 阻塞主评估 | Dify / FastGPT 文档均为开源项目文档，公开可下载 |
| 外部调研数字未经复现 | 引用出错 | 所有写进 README/简历的数字**必须本机实测** |

## 9. 里程碑

按依赖顺序，每个里程碑**独立可验收**：

| # | 里程碑 | 产出 | 验收方式 |
|---|---|---|---|
| **M1** | 中文语料入库 + 人工评估集 | 中文索引 + 30–50 条标注（含不可回答、多跳、难负例） | 索引可查；标注格式校验通过 |
| **M2** | 加权融合 + 查询路由 | `rrf_fuse(weights=)` + `route_weights` | **判据 1 在新集上通过**；结构性集不回归 |
| **M3** | 生成层（引用 + 拒答） | `qa/llm.py` `prompt.py` `generate.py` `verify.py` + `/ask` | 判据 2、3 |
| **M4** | 多轮 | `rewrite.py` + 双路检索 | 判据 4 |
| **M5** | 验收与文档 | `ACCEPTANCE_V2.md` + README 更新 | 判据 5（零回归）+ 全部数字实测 |

**M1 排在第一位是刻意的**：没有诚实的评估集，M2–M4 的每一个数字都是自欺。

## 10. 对 V1 的影响

- `/search` 行为不变；231 个测试必须全绿
- `rrf_fuse` 的默认行为不变（`weights=None`）
- `FusedHit` 新增字段带默认值，构造方式向后兼容
- `build_graph` 新增可选参数，不传时**图与 V1 结构一致**
- README 的取舍表需要新增 V2 的行（特别是"为什么从等权改加权"与
  "为什么改写成双路而不是替换"）
