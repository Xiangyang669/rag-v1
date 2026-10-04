# 多模态入库 · 阶段二：图片双通道 + PDF 入口 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让图片里的文字与视觉内容进入检索（OCR + 多模态双通道），并新增 PDF 入口覆盖扫描件与带表格的 PDF。

**Architecture:** 在阶段一的 Element 链路上加两个可替换引擎（OCR / 多模态）与一个分流合并模块。PDF 侧由 `layout.py` 做自研轻量版面分析（pdfplumber 的字符/图片/表格 bbox 聚类），产出 `Region` 列表；`PdfLoader` 把 Region 转成 Element 并对其中的图片区域跑双通道。所有外部调用（OCR 模型、VLM API、图片下载）都通过 Protocol 注入，单测全部用 fake，真实引擎只在 opt-in 的 smoke 测试里跑。

**Tech Stack:** Python 3.12 · pdfplumber 0.11.10 · pypdfium2 · Pillow 12.3 · onnxruntime 1.29 · rapidocr-onnxruntime（**唯一需新装**）· OpenAI SDK（走硅基流动）

**Spec:** `docs/superpowers/specs/2026-10-04-multimodal-ingest-design.md`
**前置：** 阶段一计划 `2026-10-04-multimodal-ingest-phase1-metadata-tables.md` 必须先全部落地

## Global Constraints

- **新增依赖只有一个**：`rapidocr-onnxruntime`（加入 `requirements.txt`）。`reportlab` 仅在**生成 PDF fixture 时**用一次，**不进** `requirements.txt`
- 本机解释器命令是 `py`；测试命令一律 `py -m pytest`（在 `rag-v1/` 下执行）
- 阶段一结束时的测试数（预期 > 150）必须全程保持全绿
- 所有单测**不得**触网、不得调用付费 API——OCR / VLM / 图片下载一律注入 fake
- 真实引擎验证放在 `@pytest.mark.smoke`，默认跳过
- 所有代码注释、docstring、错误信息用**中文**
- 降级原因码**只能**取自 `ragv1/ingest/degrade.py` 的 `ALL_CODES`
- 任何失败**必须落降级码**，不允许静默丢弃
- `degrade` 有序去重；同输入两次运行输出完全一致
- 引擎**惰性 import**：没有 OCR/VLM 需求时不许在 import 期加载模型或读 key

## 与 spec 的一处细化

spec §5 写的是 `analyze_page(page) -> list[Element]`。本计划改为返回 **`Region`** 列表：版面分析不该知道 OCR/VLM 的存在，`Region → Element` 的转换与图片通道调用属于 `loader` 的职责。这保持了「版面分析 / OCR / 多模态」三个模块互不依赖，也是 spec「模块拆分开、便于单独替换」的直接落实。

## Review Focus

以下五类输入是 spec 未逐条点名、但最可能咬到使用者的。每条都在下方任务里配了测试。

1. **VLM 把 JSON 裹在 ```json 围栏里返回** —— 朴素 `json.loads` 会失败；期望剥围栏后成功解析，而不是丢掉整条描述
2. **图片 URL 404 / 超时** —— 期望落 `image_fetch_failed` 且**整份文档入库不中断**（184 处图片，一个坏链接不能毁掉一篇文档）
3. **图片字节能取到但不是图片**（HTML 错误页、空文件）—— 期望 OCR/VLM 各自失败落码，不抛异常
4. **无文字层的页且渲染失败** —— 期望落 `pdf_render_failed` 并产出占位元素，不静默丢页
5. **加密或损坏的 PDF** —— 期望 `load_document` 不抛出，返回带降级码的元素

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `ragv1/config.py` | 修改 | OCR / VLM / 图片通道配置项 |
| `ragv1/ingest/image_source.py` | 新建 | 图片取回（URL 下载 / 本地路径）+ 缓存 |
| `ragv1/ingest/ocr.py` | 新建 | OCR 引擎抽象 + RapidOCR 默认实现 |
| `ragv1/ingest/vlm.py` | 新建 | 多模态引擎抽象 + SiliconFlow 默认实现 + JSON 解析 |
| `ragv1/ingest/image_channels.py` | 新建 | 分流规则表 + 合并格式 + `resolve_image_element` |
| `ragv1/ingest/layout.py` | 新建 | PDF 版面分析（文字层判定 + 区域聚类）→ `Region` |
| `ragv1/ingest/loader.py` | 修改 | 加图片元素识别；加 `PdfLoader` 与 `.pdf` 分派 |
| `tests/fixtures/ocr_sample.png` | 新建 | 带文字的图 |
| `tests/fixtures/flow.png` | 新建 | 流程图 |
| `tests/fixtures/table_doc.pdf` | 新建 | 带文字层与表格的小 PDF |
| `tests/test_smoke_engines.py` | 新建 | `@pytest.mark.smoke` 真实引擎验证 |
| `scripts/make_fixtures.py` | 新建 | 用 Pillow / reportlab 生成上面三份 fixture |

---

### Task 1: 引擎与图片通道配置

**Files:**
- Modify: `ragv1/config.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Test: `tests/test_channels_config.py`

**Interfaces:**
- Consumes: 无
- Produces: 模块级常量 `OCR_ENGINE / VLM_ENGINE / VLM_MODEL / OCR_MIN_CHARS / OCR_MIN_CONF / ENABLE_IMAGE / IMAGE_CACHE_DIR`；`ragv1.ingest.image_channels.ChannelConfig(min_chars: int, min_conf: float, enabled: bool)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_channels_config.py
from ragv1 import config
from ragv1.ingest.image_channels import ChannelConfig


def test_config_defaults_match_spec():
    assert config.OCR_ENGINE == "rapidocr"
    assert config.VLM_ENGINE == "siliconflow"
    assert config.VLM_MODEL == "Qwen/Qwen2.5-VL-32B-Instruct"
    assert config.OCR_MIN_CHARS == 10
    assert config.OCR_MIN_CONF == 0.5
    assert config.ENABLE_IMAGE is True
    assert config.IMAGE_CACHE_DIR.name == "images"


def test_cache_dir_lives_under_index_dir():
    assert config.IMAGE_CACHE_DIR.parent == config.INDEX_DIR


def test_channel_config_takes_defaults_from_config():
    c = ChannelConfig()
    assert c.min_chars == config.OCR_MIN_CHARS
    assert c.min_conf == config.OCR_MIN_CONF
    assert c.enabled == config.ENABLE_IMAGE


def test_engine_settings_are_env_overridable(monkeypatch):
    """可配置是需求：换引擎不该改代码"""
    monkeypatch.setenv("OCR_ENGINE", "none")
    import importlib
    importlib.reload(config)
    assert config.OCR_ENGINE == "none"
    monkeypatch.delenv("OCR_ENGINE")
    importlib.reload(config)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_channels_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragv1.ingest.image_channels'`

- [ ] **Step 3: Implement**

- `ragv1/config.py` 追加上述七个常量，值用 `os.environ.get("OCR_ENGINE", "rapidocr")` 形式，**读环境变量覆盖**；补中文注释说明各是干什么的
- `requirements.txt` 追加一行 `rapidocr-onnxruntime`
- `.gitignore` 追加 `.indexes/` 下的图片缓存路径（若 `.indexes/` 已整体忽略则跳过这条）
- `ragv1/ingest/image_channels.py` 先只放 `ChannelConfig` dataclass（默认值取自 `config`）与模块 docstring，函数在 Task 5 补

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_channels_config.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 安装新依赖并验证 import 期不加载模型**

```bash
py -m pip install rapidocr-onnxruntime
py -c "import time; t=time.time(); import ragv1.ingest.image_channels; print('import 耗时', round(time.time()-t,3), '秒')"
```
Expected: import 耗时 < 2 秒（证明惰性 import 成立）

- [ ] **Step 6: Run the whole suite & commit**

Run: `py -m pytest -q`
Expected: 阶段一基线 + 4，全绿

```bash
git add ragv1/config.py ragv1/ingest/image_channels.py requirements.txt .gitignore tests/test_channels_config.py
git commit -m "feat(ingest): OCR/多模态/图片通道配置项，新增 rapidocr 依赖"
```

---

### Task 2: 图片取回与缓存

**Files:**
- Create: `ragv1/ingest/image_source.py`
- Test: `tests/test_image_source.py`

**Interfaces:**
- Consumes: `ragv1.ingest.degrade`
- Produces:
  - `ImageBytes(data: bytes | None, path: str | None, degrade: tuple[str, ...] = ())`
  - `resolve_image_bytes(ref: str, base_dir: Path, cache_dir: Path, fetcher: Callable[[str], bytes] | None = None) -> ImageBytes`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_image_source.py
from pathlib import Path

import pytest

from ragv1.ingest import degrade
from ragv1.ingest.image_source import resolve_image_bytes

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"


def test_remote_url_is_downloaded_and_cached(tmp_path):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return PNG

    out = resolve_image_bytes("http://x/y.png", tmp_path, tmp_path / "cache", fetcher=fake_fetch)
    assert out.data == PNG
    assert out.degrade == ()
    assert Path(out.path).exists() and Path(out.path).read_bytes() == PNG

    # 第二次命中缓存，不再下载
    out2 = resolve_image_bytes("http://x/y.png", tmp_path, tmp_path / "cache", fetcher=fake_fetch)
    assert out2.data == PNG
    assert len(calls) == 1, "缓存未命中，重复下载了"


def test_local_relative_path_resolves_against_base_dir(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    out = resolve_image_bytes("a.png", tmp_path, tmp_path / "cache")
    assert out.data == PNG and out.degrade == ()


def test_missing_local_file_degrades(tmp_path):
    """Review Focus #2 的本地版：坏引用不能中断入库"""
    out = resolve_image_bytes("nope.png", tmp_path, tmp_path / "cache")
    assert out.data is None
    assert degrade.IMAGE_MISSING in out.degrade


def test_fetch_failure_degrades_and_does_not_raise(tmp_path):
    """Review Focus #2：URL 404 / 超时"""
    def boom(url):
        raise OSError("404")

    out = resolve_image_bytes("http://x/y.png", tmp_path, tmp_path / "cache", fetcher=boom)
    assert out.data is None
    assert degrade.IMAGE_FETCH_FAILED in out.degrade


def test_cache_write_failure_still_returns_bytes(tmp_path):
    """缓存目录不可写时，数据仍要能用——缓存是优化不是依赖"""
    def fake_fetch(url):
        return PNG

    blocked = tmp_path / "blocked"
    blocked.write_text("我是文件不是目录", encoding="utf-8")
    out = resolve_image_bytes("http://x/y.png", tmp_path, blocked, fetcher=fake_fetch)
    assert out.data == PNG
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_image_source.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `ragv1/ingest/image_source.py`**

要点：
- `ref` 以 `http://` / `https://` 开头 → 走 `fetcher`；`fetcher` 缺省为用 `urllib.request.urlopen` 的实现（**带超时**，缺省 10 秒）
- 缓存文件名：`hashlib.sha1(ref.encode("utf-8")).hexdigest()[:16]` + 原后缀；缓存命中直接读盘
- 非 URL → 相对 `base_dir` 解析；不存在落 `IMAGE_MISSING`
- 任何异常都转成降级码，**绝不向上抛**；缓存写入失败只记 debug，仍返回 bytes
- 返回的 `path` 字段是本地缓存路径（命中或写入成功时）

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_image_source.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: Run the whole suite & commit**

Run: `py -m pytest -q`

```bash
git add ragv1/ingest/image_source.py tests/test_image_source.py
git commit -m "feat(ingest): 图片取回与本地缓存，失败降级不中断"
```

---

### Task 3: OCR 引擎

**Files:**
- Create: `ragv1/ingest/ocr.py`
- Test: `tests/test_ocr.py`

**Interfaces:**
- Consumes: `ragv1.config`
- Produces:
  - `OcrResult(text: str, confidence: float, lines: tuple[str, ...] = ())`
  - `class OcrEngine(Protocol): def extract(self, image: bytes) -> OcrResult`
  - `class RapidOcrEngine(min_conf: float = OCR_MIN_CONF)` — 惰性持有 rapidocr 实例
  - `build_ocr_engine(name: str, min_conf: float = OCR_MIN_CONF) -> OcrEngine | None`（`"none"` → `None`）

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ocr.py
from ragv1.ingest.ocr import OcrResult, build_ocr_engine


def test_build_returns_none_for_none_engine():
    assert build_ocr_engine("none") is None


def test_build_returns_none_for_unknown_engine():
    """配错的引擎名不该让入库崩掉，而是干净地降级"""
    assert build_ocr_engine("no-such-engine") is None


def test_rapidocr_engine_is_lazy_until_first_call(monkeypatch):
    """import 期不许加载模型"""
    eng = build_ocr_engine("rapidocr")
    assert eng is not None
    assert getattr(eng, "_impl", None) is None      # 还没实例化 rapidocr


def test_extract_wraps_engine_output_into_result(monkeypatch):
    eng = build_ocr_engine("rapidocr")

    class FakeImpl:
        def __call__(self, img):
            return ([[None, "你好", 0.98], [None, "世界", 0.90]], None)

    monkeypatch.setattr(eng, "_impl", FakeImpl(), raising=False)
    r = eng.extract(b"whatever")
    assert isinstance(r, OcrResult)
    assert r.text == "你好\n世界"
    assert r.lines == ("你好", "世界")
    assert abs(r.confidence - 0.94) < 1e-6          # 逐行均值


def test_extract_on_engine_exception_returns_empty_result(monkeypatch):
    """Review Focus #3：拿到非图片字节时不能抛"""
    eng = build_ocr_engine("rapidocr")

    class BoomImpl:
        def __call__(self, img):
            raise ValueError("不是图片")

    monkeypatch.setattr(eng, "_impl", BoomImpl(), raising=False)
    r = eng.extract(b"<html>404</html>")
    assert r.text == "" and r.confidence == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_ocr.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `ragv1/ingest/ocr.py`**

要点：
- `RapidOcrEngine.extract` 首次调用时才 `from rapidocr_onnxruntime import RapidOCR` 并构造实例（惰性）；用 `functools.cached_property` 或显式 `_impl` 属性
- rapidocr 返回 `(result, elapse)`，`result` 为 `[[box, text, score], ...]`——**过滤掉 `result` 为 `None` 或空列表的情况**（无文字时它返回 `None`）
- `confidence` = 各行 score 的均值；无行时 `0.0`
- 任何异常 → 返回 `OcrResult("", 0.0)`（**降级码由调用方 `image_channels` 依据「文本为空」判定**，引擎自己不管码）

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_ocr.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: Run the whole suite & commit**

Run: `py -m pytest -q`

```bash
git add ragv1/ingest/ocr.py tests/test_ocr.py
git commit -m "feat(ingest): OCR 引擎抽象与 RapidOCR 惰性实现"
```

---

### Task 4: 多模态引擎

**Files:**
- Create: `ragv1/ingest/vlm.py`
- Test: `tests/test_vlm.py`

**Interfaces:**
- Consumes: `ragv1.config`
- Produces:
  - `VisionResult(image_type: str, content: str, data: str = "", relations: str = "", raw: str = "")`
  - `class VisionEngine(Protocol): def describe(self, image: bytes) -> VisionResult`
  - `parse_vision_json(raw: str) -> VisionResult | None`
  - `class SiliconFlowVision(model: str = VLM_MODEL, api_key: str | None = None, client=None)`
  - `build_vision_engine(name: str) -> VisionEngine | None`
  - `VISION_PROMPT: str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vlm.py
from ragv1.ingest.vlm import build_vision_engine, parse_vision_json

GOOD = '{"image_type":"流程图","content":"三步流程","data":"","relations":"A→B→C"}'


def test_parse_plain_json():
    r = parse_vision_json(GOOD)
    assert r.image_type == "流程图" and r.content == "三步流程" and r.relations == "A→B→C"


def test_parse_strips_markdown_json_fence():
    """Review Focus #1：VLM 爱把 JSON 裹进 ```json 围栏"""
    r = parse_vision_json(f"```json\n{GOOD}\n```")
    assert r is not None and r.content == "三步流程"


def test_parse_extracts_json_from_surrounding_prose():
    r = parse_vision_json(f"好的，结果如下：\n{GOOD}\n希望有帮助。")
    assert r is not None and r.image_type == "流程图"


def test_parse_returns_none_on_garbage():
    assert parse_vision_json("这不是 JSON") is None


def test_parse_missing_keys_defaults_to_empty():
    r = parse_vision_json('{"image_type":"图表"}')
    assert r is not None and r.content == "" and r.data == ""


def test_build_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    assert build_vision_engine("siliconflow") is None


def test_build_returns_none_for_unknown_engine():
    assert build_vision_engine("no-such-engine") is None


def test_describe_sends_data_url_and_parses_reply():
    import base64
    from ragv1.ingest.vlm import SiliconFlowVision

    captured = {}

    class FakeCompletions:
        def create(self, **kw):
            captured.update(kw)
            class Msg:
                content = GOOD
            class Choice:
                message = Msg()
            class Resp:
                choices = [Choice()]
            return Resp()

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    eng = SiliconFlowVision(client=FakeClient())
    r = eng.describe(b"\x89PNG-fake")
    assert r.content == "三步流程"
    assert "data:image/png;base64," in captured["messages"][0]["content"][1]["image_url"]["url"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_vlm.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `ragv1/ingest/vlm.py`**

要点：
- `parse_vision_json`：先剥 ```` ```json ... ``` ```` 围栏（正则匹配最外层围栏）；再用 `raw.find("{")` / `raw.rfind("}")` 截出 JSON 子串后 `json.loads`；失败返回 `None`。**不要**用贪婪正则找 JSON
- `VisionResult.raw` 保留原始返回文本
- `SiliconFlowVision`：`api_key` 缺省读 `SILICONFLOW_API_KEY`；`client` 用 `openai.OpenAI(base_url="https://api.siliconflow.cn/v1", api_key=...)`；`describe` 把图片编码成 `data:image/png;base64,...` 的 data URL，以 `[{"type":"text","text":VISION_PROMPT},{"type":"image_url","image_url":{"url":...}}]` 形式发送
- `VISION_PROMPT`：要求**只输出 JSON**，字段为 `image_type`（流程图|图表|截图|照片|其他）/`content`/`data`/`relations`，中文说明
- `describe` 的异常处理：`parse_vision_json` 失败 → 返回 `VisionResult("其他", raw[:200], raw=raw)`，**调用方据此落 `vlm_bad_json` 但保留原文**

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_vlm.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: Run the whole suite & commit**

Run: `py -m pytest -q`

```bash
git add ragv1/ingest/vlm.py tests/test_vlm.py
git commit -m "feat(ingest): 多模态引擎抽象与 SiliconFlow 实现，JSON 宽松解析"
```

---

### Task 5: 图片双通道分流与合并

**Files:**
- Modify: `ragv1/ingest/image_channels.py`
- Test: `tests/test_image_channels.py`

**Interfaces:**
- Consumes: `ragv1.types.Element`、`ragv1.ingest.{ocr, vlm, image_source, degrade}`、`ChannelConfig`
- Produces:
  - `resolve_image_element(el: Element, ocr: OcrEngine | None, vlm: VisionEngine | None, *, base_dir: Path, cache_dir: Path, fetcher=None, cfg: ChannelConfig = ChannelConfig()) -> Element`
  - `merge_image_text(alt: str, ref: str, ocr_text: str, vision: VisionResult | None) -> str`

- [ ] **Step 1: Write the failing test — 分流规则表逐行覆盖**

```python
# tests/test_image_channels.py
from pathlib import Path

from ragv1.ingest import degrade
from ragv1.ingest.ocr import OcrResult
from ragv1.ingest.vlm import VisionResult
from ragv1.ingest.image_channels import ChannelConfig, merge_image_text, resolve_image_element
from ragv1.types import Element

PNG = b"\x89PNG\r\n\x1a\nbody"


class FakeOcr:
    def __init__(self, text, conf=0.9):
        self.r = OcrResult(text, conf, tuple(text.splitlines()) if text else ())

    def extract(self, image):
        return self.r


class FakeVlm:
    def __init__(self, image_type="截图", content="描述", data="", relations="", raw=None):
        self.r = VisionResult(image_type, content, data, relations, raw or content)

    def describe(self, image):
        return self.r


def _el(ref="a.png", alt="示意图", base=None):
    return Element(kind="image", order=0, text=alt, image_ref=ref)


def _run(tmp_path, ocr, vlm, cfg=ChannelConfig()):
    (tmp_path / "a.png").write_bytes(PNG)
    return resolve_image_element(
        _el(), ocr, vlm, base_dir=tmp_path, cache_dir=tmp_path / "c",
        fetcher=lambda u: PNG, cfg=cfg,
    )


def test_row1_ocr_only_when_vlm_unavailable(tmp_path):
    out = _run(tmp_path, FakeOcr("服务状态正常"), None)
    assert "服务状态正常" in out.text
    assert degrade.VLM_UNAVAILABLE in out.degrade


def test_row2_text_screenshot_uses_ocr_only(tmp_path):
    out = _run(tmp_path, FakeOcr("报错：连接超时"), FakeVlm(image_type="截图", content="一个报错弹窗"))
    assert "报错：连接超时" in out.text
    assert "一个报错弹窗" not in out.text          # 纯文字截图不留多模态描述
    assert out.degrade == ()


def test_row3_chart_produces_both_channels(tmp_path):
    out = _run(tmp_path, FakeOcr("开始 结束"),
               FakeVlm(image_type="流程图", content="三步流程", relations="开始→处理→结束"))
    assert "开始 结束" in out.text                 # OCR 通道
    assert "三步流程" in out.text                  # 多模态通道（两者兼有则都产出）
    assert "开始→处理→结束" in out.text
    assert out.degrade == ()


def test_row4_visual_only_uses_vlm_only(tmp_path):
    out = _run(tmp_path, FakeOcr(""), FakeVlm(image_type="照片", content="一台服务器机柜"))
    assert "一台服务器机柜" in out.text
    assert "文字:" not in out.text
    assert out.degrade == ()


def test_row5_neither_available_keeps_placeholder(tmp_path):
    """不静默丢弃：两个通道都不可用时仍留占位文本 + 说明"""
    out = _run(tmp_path, None, None)
    assert out.text.strip() != ""
    assert degrade.OCR_EMPTY in out.degrade and degrade.VLM_UNAVAILABLE in out.degrade


def test_ocr_empty_but_vlm_unavailable_is_recorded(tmp_path):
    out = _run(tmp_path, FakeOcr(""), None)
    assert out.text.strip() != ""
    assert set(out.degrade) >= {degrade.OCR_EMPTY, degrade.VLM_UNAVAILABLE}


def test_bad_vlm_json_keeps_raw_text(tmp_path):
    """Review Focus #1：解析失败也要把原文留下"""
    vlm = FakeVlm(image_type="其他", content="", raw="```json\n{坏掉的\n```")
    vlm.r = VisionResult("其他", "", raw=vlm.r.raw)
    out = _run(tmp_path, FakeOcr(""), vlm)
    assert "坏掉的" in out.text


def test_fetch_failure_still_yields_element_with_degrade(tmp_path):
    out = resolve_image_element(
        _el(ref="http://x/nope.png"), FakeOcr("x"), FakeVlm(),
        base_dir=tmp_path, cache_dir=tmp_path / "c",
        fetcher=lambda u: (_ for _ in ()).throw(OSError("404")),
    )
    assert degrade.IMAGE_FETCH_FAILED in out.degrade
    assert out.text.strip() != ""


def test_merge_format_only_includes_non_empty_parts():
    text = merge_image_text("", "a.png", "文字内容", None)
    assert text.startswith("[图片]")
    assert "文字: 文字内容" in text
    assert "类型:" not in text and "关系:" not in text


def test_resolved_element_carries_image_path(tmp_path):
    out = _run(tmp_path, FakeOcr("x"), None)
    assert out.image_path and Path(out.image_path).exists()
    assert out.image_ref == "a.png"
    assert out.kind == "image"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_image_channels.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_image_element'`

- [ ] **Step 3: Implement in `ragv1/ingest/image_channels.py`**

严格按 spec §6.2 的规则表实现，**五行分支都要写出来**（测试逐行钉住）：

| OCR 有料¹ | VLM 可用 | VLM 判定 | 输出 |
|---|---|---|---|
| 是 | 否 | — | OCR 文本 + `vlm_unavailable` |
| 是 | 是 | 纯文字截图 | 仅 OCR |
| 是 | 是 | 图表/流程图/照片 | OCR + 多模态（都产出）|
| 否 | 是 | — | 仅多模态 |
| 否 | 否 | — | 占位 + `ocr_empty` + `vlm_unavailable` |

¹ 「有料」= `len(text.strip()) >= cfg.min_chars` 且 `confidence >= cfg.min_conf`。

补充规则：
- `cfg.enabled is False` → 完全不解析，text 保持 alt/`[图片] ref`，落 `vlm_unavailable`（表示通道未启用）
- OCR 抛异常 → `ocr_failed`；VLM 抛异常 → `vlm_failed`
- `VisionResult.content` 为空但 `raw` 非空 → 视为 `vlm_bad_json`，`raw` 当 content 用
- `image_type` 属于 `{"图表", "流程图", "照片"}` 视为「视觉为主」；`{"截图", "其他"}` 视为「文字为主」
- `merge_image_text` 按 spec §6.2 的格式拼，**只含非空段**，段序为：`[图片] 标题 / 类型 / 内容 / 数据 / 关系 / 文字`
- 返回新 `Element`（`dataclasses.replace`），保留 `order` / `heading_path` / `page` / `bbox` / `image_ref`，`text` 换成合并结果，`degrade` 用 `degrade.normalize` 去重

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_image_channels.py -v`
Expected: PASS（11 passed）

- [ ] **Step 5: Run the whole suite & commit**

Run: `py -m pytest -q`

```bash
git add ragv1/ingest/image_channels.py tests/test_image_channels.py
git commit -m "feat(ingest): 图片双通道分流规则与合并格式"
```

---

### Task 6: MarkdownLoader 接入图片元素

**Files:**
- Modify: `ragv1/ingest/loader.py`
- Test: `tests/test_loader.py`（追加）

**Interfaces:**
- Consumes: `ragv1.ingest.image_channels.resolve_image_element`
- Produces: `markdown_elements(text, doc_id, *, ocr=None, vlm=None, base_dir=None, cache_dir=None, fetcher=None, cfg=None) -> list[Element]`（新参数全部可选，**不传时行为与阶段一完全一致**——这保证阶段一测试不受影响）

- [ ] **Step 1: Write the failing test**

```python
# 追加到 tests/test_loader.py
import pytest
from ragv1.ingest import degrade
from ragv1.ingest.ocr import OcrResult


class _Ocr:
    def extract(self, image):
        return OcrResult("图里的字", 0.9, ("图里的字",))


def test_image_line_becomes_image_element(tmp_path):
    (tmp_path / "a.png").write_bytes(b"\x89PNG\r\n\x1a\nbody")
    md = "# H\n\n前段。\n\n![示意图](a.png)\n\n后段。\n"
    els = markdown_elements(md, doc_id="d.md", ocr=_Ocr(), base_dir=tmp_path,
                            cache_dir=tmp_path / "c")
    assert [e.kind for e in els] == ["text", "image", "text"]
    assert "图里的字" in els[1].text
    assert els[1].image_ref == "a.png"
    assert [e.order for e in els] == [0, 1, 2]


def test_image_inside_code_fence_is_not_an_image(tmp_path):
    md = "# H\n\n```md\n![x](y.png)\n```\n"
    els = markdown_elements(md, doc_id="d.md", base_dir=tmp_path, cache_dir=tmp_path / "c")
    assert [e.kind for e in els] == ["text"]
    assert "![x](y.png)" in els[0].text


def test_inline_image_stays_in_text(tmp_path):
    """spec §6.1：行内图片不拆开，保留上下文"""
    md = "# H\n\n前面文字 ![x](y.png) 后面文字\n"
    els = markdown_elements(md, doc_id="d.md", base_dir=tmp_path, cache_dir=tmp_path / "c")
    assert [e.kind for e in els] == ["text"]
    assert "![x](y.png)" in els[0].text


def test_reference_style_image_degrades_and_stays_in_text(tmp_path):
    md = "# H\n\n![x][id]\n\n[id]: y.png\n"
    els = markdown_elements(md, doc_id="d.md", base_dir=tmp_path, cache_dir=tmp_path / "c")
    assert [e.kind for e in els] == ["text"]
    assert degrade.IMAGE_REF_UNRESOLVED in els[0].degrade


def test_no_engines_configured_keeps_phase1_behavior(tmp_path):
    """不传 ocr/vlm 时，图片行仍是普通正文 —— 阶段一测试因此不受影响"""
    md = "# H\n\n![示意图](a.png)\n"
    els = markdown_elements(md, doc_id="d.md")
    assert [e.kind for e in els] == ["text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_loader.py -v`
Expected: FAIL — `test_image_line_becomes_image_element` 失败（图片行还在正文里）

- [ ] **Step 3: Implement in `ragv1/ingest/loader.py`**

要点：
- 行级判定**整行** `strip()` 后匹配 `^!\[(?P<alt>[^\]]*)\]\((?P<ref>[^)]+)\)$` 才算图片；匹配 `^!\[[^\]]*\]\[[^\]]*\]$` 的引用式图片 → 该行走「不识别」分支并把 `IMAGE_REF_UNRESOLVED` 记到当前正文段上
- 图片行是正文的断开点：flush 当前正文 → 产出 `Element(kind="image", text=alt or "[图片] ref", image_ref=ref, order=...)`
- **仅当 `cfg.enabled` 且至少给了 `ocr` / `vlm` 之一时**才调 `resolve_image_element`；否则图片元素保持 `text=alt or "[图片] ref"`、`degrade=()`（**这就是「不传引擎 = 阶段一行为」的实现**）
- 图片的 `base_dir` 缺省取 `Path.cwd()`；调用方（`load_document`）负责传文档所在目录

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_loader.py -v`
Expected: PASS

- [ ] **Step 5: Run the whole suite & commit**

Run: `py -m pytest -q`
Expected: 全绿，含阶段一全部测试

```bash
git add ragv1/ingest/loader.py tests/test_loader.py
git commit -m "feat(ingest): MarkdownLoader 识别图片元素并接入双通道"
```

---

### Task 7: PDF 版面分析

**Files:**
- Create: `ragv1/ingest/layout.py`
- Test: `tests/test_layout.py`

**Interfaces:**
- Consumes: `ragv1.ingest.degrade`
- Produces:
  - `Region(kind: str, page: int, bbox: tuple[float,float,float,float], text: str = "", table_rows: tuple[tuple[str, ...], ...] = (), image_bytes: bytes | None = None, degrade: tuple[str, ...] = ())`
  - `has_text_layer(page) -> bool`
  - `analyze_page(page, page_no: int, render_page: Callable[[], bytes] | None = None) -> list[Region]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_layout.py
from ragv1.ingest import degrade
from ragv1.ingest.layout import Region, analyze_page, has_text_layer


class FakeChar:
    def __init__(self, text, x0, top, x1, bottom):
        self.text, self.x0, self.top, self.x1, self.bottom = text, x0, top, x1, bottom


class FakePage:
    """pdfplumber Page 的最小替身：只暴露 analyze_page 用到的三个属性。"""

    def __init__(self, chars=(), images=(), tables=()):
        self.chars = list(chars)
        self.images = list(images)
        self._tables = list(tables)

    def find_tables(self):
        return self._tables


class FakeTable:
    def __init__(self, bbox, rows):
        self.bbox, self.rows = bbox, rows


def test_has_text_layer_false_on_scanned_page():
    assert has_text_layer(FakePage(chars=[])) is False


def test_has_text_layer_true_when_chars_present():
    assert has_text_layer(FakePage(chars=[FakeChar("a", 0, 0, 1, 1)])) is True


def test_scanned_page_is_rendered_and_marked():
    """Review Focus #4：无文字层的页要走渲染"""
    regions = analyze_page(FakePage(), 1, render_page=lambda: b"\x89PNG-page")
    assert len(regions) == 1
    assert regions[0].kind == "image" and regions[0].image_bytes == b"\x89PNG-page"
    assert degrade.PDF_PAGE_NO_TEXT_LAYER in regions[0].degrade


def test_render_failure_yields_placeholder_not_silence():
    """Review Focus #4：渲染失败也不能静默丢页"""
    def boom():
        raise RuntimeError("渲染挂了")

    regions = analyze_page(FakePage(), 1, render_page=boom)
    assert len(regions) == 1
    assert degrade.PDF_RENDER_FAILED in regions[0].degrade


def test_text_page_yields_text_region():
    page = FakePage(chars=[FakeChar("h", 0, 0, 10, 10), FakeChar("i", 11, 0, 20, 10)])
    regions = analyze_page(page, 1)
    assert [r.kind for r in regions] == ["text"]
    assert "hi" in regions[0].text
    assert regions[0].page == 1


def test_table_region_is_separated_from_text():
    page = FakePage(
        chars=[FakeChar("正", 0, 100, 10, 110), FakeChar("文", 11, 100, 20, 110)],
        tables=[FakeTable((0, 0, 100, 50), [["a", "b"], ["1", "2"]])],
    )
    regions = analyze_page(page, 1)
    kinds = sorted(r.kind for r in regions)
    assert kinds == ["table", "text"]
    table = next(r for r in regions if r.kind == "table")
    assert table.table_rows == (("a", "b"), ("1", "2"))


def test_image_region_is_separated(tmp_path):
    page = FakePage(
        chars=[FakeChar("字", 0, 200, 10, 210)],
        images=[{"x0": 0, "top": 0, "x1": 50, "bottom": 50, "stream": b"IMG"}],
    )
    regions = analyze_page(page, 1)
    imgs = [r for r in regions if r.kind == "image"]
    assert len(imgs) == 1
    assert imgs[0].image_bytes is not None or degrade.PDF_RENDER_FAILED in imgs[0].degrade


def test_regions_are_ordered_top_to_bottom():
    page = FakePage(chars=[FakeChar("下", 0, 500, 10, 510), FakeChar("上", 0, 10, 10, 20)])
    texts = [r.text for r in analyze_page(page, 1) if r.kind == "text"]
    assert "".join(texts).startswith("上")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_layout.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `ragv1/ingest/layout.py`**

要点：
- `has_text_layer(page)`：`bool(page.chars)` —— 简单、确定，注释说明为什么不看字符数阈值（空白字符也算文字层，靠后续质量判定）
- `analyze_page` 流程：
  1. **无文字层** → 调 `render_page()`：成功则产出整页 image Region（落 `PDF_PAGE_NO_TEXT_LAYER`）；异常则产出占位 text Region（落 `PDF_RENDER_FAILED`）并返回
  2. **有文字层** → 收集 `page.find_tables()` 的 bbox 与 `page.images` 的 bbox，按 **top 排序**
  3. `chars` 按 `top` 聚成行（同一行判定：`top` 差值 < 行高的一半），行按 top 排序
  4. 每行按 x 排序拼成文本；逐行判定它落在哪个区域 bbox 内
  5. 产出的 Region **按 bbox.top 排序**（阅读顺序）
- 表格行的文本内容取 `table.rows`（pdfplumber 的 `Table.rows` 提供 `.cells`，用 `[[c or "" for c in row.cells] for row in table.rows]`）；取不到时落 `PDF_TABLE_BBOX_MISSING`
- 图片字节：优先从 `image["stream"]` 取（pdfplumber 的 `page.images` 元素含 `stream`）；取不到则落 `PDF_RENDER_FAILED`，由 loader 决定是否整页渲染兜底

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_layout.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: Run the whole suite & commit**

Run: `py -m pytest -q`

```bash
git add ragv1/ingest/layout.py tests/test_layout.py
git commit -m "feat(ingest): PDF 版面分析（文字层判定 + 区域聚类）"
```

---

### Task 8: PdfLoader

**Files:**
- Modify: `ragv1/ingest/loader.py`
- Test: `tests/test_loader.py`（追加）

**Interfaces:**
- Consumes: `ragv1.ingest.layout.analyze_page`、`ragv1.ingest.table.normalize_table`、`resolve_image_element`
- Produces: `load_document(path, doc_id, **engines) -> list[Element]` 支持 `.pdf`

- [ ] **Step 1: Write the failing test**

```python
# 追加到 tests/test_loader.py
def test_load_pdf_returns_elements(tmp_path):
    """用假 page 对象注入，不依赖真 PDF 文件"""
    from ragv1.ingest import loader

    pages = [FakePage_Layout(chars=[FakeChar_Layout("你好", 0, 0, 20, 10)])]
    els = loader.pdf_elements(_FakePdf(pages), doc_id="a.pdf")
    assert [e.kind for e in els] == ["text"]
    assert "你好" in els[0].text
    assert els[0].page == 1


def test_pdf_page_numbers_are_one_based_and_ordered(tmp_path):
    pages = [
        _FakePage([_C("第一页", 0, 0, 20, 10)]),
        _FakePage([_C("第二页", 0, 0, 20, 10)]),
    ]
    els = loader.pdf_elements(_FakePdf(pages), doc_id="a.pdf")
    assert [e.page for e in els] == [1, 2]
    assert [e.order for e in els] == [0, 1]


def test_corrupt_pdf_degrades_instead_of_raising(tmp_path):
    """Review Focus #5：损坏/加密的 PDF 不能抛出"""
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 not really a pdf")
    els = load_document(bad, doc_id="bad.pdf")
    assert isinstance(els, list)          # 不抛异常即可，允许为空
```

`_FakePdf` / `_FakePage` / `_C` 为测试内的最小替身，暴露 `pages` 与 `chars`。

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_loader.py -v`
Expected: FAIL — `AttributeError: module 'ragv1.ingest.loader' has no attribute 'pdf_elements'`

- [ ] **Step 3: Implement in `ragv1/ingest/loader.py`**

- 新增 `pdf_elements(pdf, doc_id, *, ocr=None, vlm=None, cache_dir=None, fetcher=None, cfg=None) -> list[Element]`：遍历 `pdf.pages`（1 起编号），每页调 `analyze_page`，把 Region 转成 Element：
  - `text` Region → `Element(kind="text", ...)`
  - `table` Region → `table.normalize_table(["| " + " | ".join(r) + " |" for r in rows] + 分隔行)`；返回 `None` → 降级成 text Element 并落 `TABLE_UNSTRUCTURED`
  - `image` Region → 有 `image_bytes` 就走 `resolve_image_element`（**走同一套双通道，与 Markdown 侧共用**）；无字节且 Region 已带降级码则沿用
- `load_document` 增加 `.pdf` 分支：用 `pdfplumber.open(path)` 打开，**整体包 try/except**——打不开时返回 `[]` 并打印警告（Review Focus #5）
- `order` 跨页连续递增；`heading_path` 对 PDF 一律为空元组

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_loader.py -v`
Expected: PASS

- [ ] **Step 5: Run the whole suite & commit**

Run: `py -m pytest -q`

```bash
git add ragv1/ingest/loader.py tests/test_loader.py
git commit -m "feat(ingest): PdfLoader（版面分析 → Element，扫描页走双通道）"
```

---

### Task 9: Fixtures 与真实引擎 smoke

**Files:**
- Create: `scripts/make_fixtures.py`、`tests/fixtures/ocr_sample.png`、`tests/fixtures/flow.png`、`tests/fixtures/table_doc.pdf`
- Create: `tests/test_smoke_engines.py`
- Modify: `pytest.ini`（注册 `smoke` marker）

**Interfaces:**
- Consumes: `build_ocr_engine`、`build_vision_engine`、`load_document`
- Produces: 一份**实测输出**，写进验收报告

- [ ] **Step 1: Write `scripts/make_fixtures.py`**

- `ocr_sample.png`：Pillow 画白底黑字，内容含中文与数字（例如 `服务状态正常 2026-10-04 CPU 87%`），字号足够大（≥28）
- `flow.png`：Pillow 画三个方框 + 箭头 + 标签（`开始` / `处理` / `结束`），白底黑线
- `table_doc.pdf`：用 **reportlab** 生成两页——第 1 页有标题、正文段落与一张 3 列表格（带边框）；第 2 页只有一段文字（用来验 `page` 字段是 1/2）

Run: `py -m pip install reportlab && py scripts/make_fixtures.py`
（`reportlab` **只在生成 fixture 时用**，不写进 `requirements.txt`）

- [ ] **Step 2: Write failing smoke tests**

```python
# tests/test_smoke_engines.py
"""真实引擎验证。默认跳过：py -m pytest -m smoke -v"""
import os
from pathlib import Path

import pytest

from ragv1.ingest.loader import load_document
from ragv1.ingest.ocr import build_ocr_engine
from ragv1.ingest.vlm import build_vision_engine

pytestmark = pytest.mark.smoke

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.skipif(build_ocr_engine("rapidocr") is None, reason="RapidOCR 不可用")
def test_ocr_reads_text_from_screenshot():
    r = build_ocr_engine("rapidocr").extract((FIXTURES / "ocr_sample.png").read_bytes())
    print("OCR 实测输出：", repr(r.text), "置信度", r.confidence)
    assert "服务状态" in r.text.replace(" ", "")
    assert r.confidence > 0.5


@pytest.mark.skipif(
    build_vision_engine("siliconflow") is None, reason="未配置 SILICONFLOW_API_KEY"
)
def test_vlm_describes_flowchart():
    v = build_vision_engine("siliconflow").describe((FIXTURES / "flow.png").read_bytes())
    print("VLM 实测输出：", v)
    assert v.image_type in {"流程图", "图表"}
    assert v.content


def test_pdf_fixture_yields_paged_elements():
    els = load_document(FIXTURES / "table_doc.pdf", doc_id="table_doc.pdf")
    pages = {e.page for e in els}
    assert 1 in pages and 2 in pages
    print("PDF 实测元素：", [(e.kind, e.page, e.text[:30]) for e in els])
```

- [ ] **Step 3: Register the marker and verify smoke is skipped by default**

`pytest.ini` 追加：

```ini
markers =
    smoke: 需要真实 OCR / 多模态引擎或付费 API，默认跳过
```

Run: `py -m pytest -q`
Expected: 全绿，且 smoke 用例被 **skipped**（不是失败）

- [ ] **Step 4: Run the smoke suite for real**

Run: `py -m pytest -m smoke -v -s`
Expected: OCR 与 PDF 两条 PASS。VLM 那条**需要 `SILICONFLOW_API_KEY`**——没有 key 时它 skip，**如实说明，不要伪造结果**

**把 `-s` 打印出的实际文字与描述记下来**——它们进验收报告，而不是写「跑通了」。

- [ ] **Step 5: Commit**

```bash
git add scripts/make_fixtures.py tests/fixtures tests/test_smoke_engines.py pytest.ini
git commit -m "test: 三份 fixture（带字图/流程图/表格 PDF）与真实引擎 smoke 测试"
```

---

### Task 10: 阶段二验收——真实语料全量跑一次

**Files:**
- Modify: `README.md`
- Modify: `scripts/verify_tables.py` → 更名/扩展为 `scripts/verify_ingest.py`

**Interfaces:**
- Consumes: `build_corpus`、`FtsStore`
- Produces: 一份**实测数字**

- [ ] **Step 1: Extend the verification script**

在阶段一脚本基础上追加：`kind` 分布里新增 `image` 一类；统计 `degrade` 原因码分布；统计图片块数、其中 OCR 贡献文字 / 多模态贡献描述 / 两者兼有的各多少。

- [ ] **Step 2: Run it against the real corpus**

Run: `py scripts/verify_ingest.py`

⚠️ 语料有 **184 处远程图片**（全在 `raw.githubusercontent.com`）。这一步会真实下载它们——**先确认网络可用再跑**；首次会慢。

- [ ] **Step 3: 核对验收标准**

对照 spec §12：

- [ ] 含表格文档：大表被切成多块，每块带表头，无被切断的行
- [ ] 带文字的图：OCR 文本进了 chunk
- [ ] 流程图：多模态描述含类型/内容/关系（依赖 Step 4 的 smoke 结果）
- [ ] 每个 chunk 带 `doc_id` + `page`/`order`；图片 chunk 带 `image_ref` 与 `image_path`
- [ ] 降级路径都有测试；无静默丢弃
- [ ] `py -m pytest -q` 全绿，`py -m pytest -m smoke -v` 的实测结果已记录

- [ ] **Step 4: Update README**

「实测数字」表加阶段二结果；「已知限制」补两条：自研版面分析在多栏 PDF 上可能切歪；图片需联网下载并缓存。

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_ingest.py README.md
git commit -m "chore: 阶段二验收脚本与真实语料实测数字"
```

---

## 阶段二完成后的状态

- 图片走 OCR + 多模态双通道，按内容分流，兼有则都产出，按 `order` 恢复阅读顺序
- PDF 有入口：文字层页做版面分析，扫描页渲染后走双通道
- 引擎全部可配置、可注入；单测不触网，真实引擎有 opt-in smoke
- 三份 fixture 各有可运行测试
