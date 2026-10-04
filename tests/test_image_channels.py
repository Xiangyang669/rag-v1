"""图片双通道的分流规则表与合并格式。

分流规则是**一张显式的表**（见 resolve_image_element），本文件逐行钉住它。
规则表的测试用 ChannelConfig(min_chars=1) 显式放宽阈值——它们考的是分支，
不是阈值；阈值另有 test_min_chars_threshold_decides_text_vs_visual 专门覆盖。
"""

from pathlib import Path

from ragv1.ingest import degrade
from ragv1.ingest.image_channels import (
    ChannelConfig,
    merge_image_text,
    resolve_image_element,
)
from ragv1.ingest.ocr import OcrResult
from ragv1.ingest.vlm import VisionResult
from ragv1.types import Element

PNG = b"\x89PNG\r\n\x1a\nbody"

# 规则表测试统一用这个：把「有料」的判据降到 1 个字，
# 这样分支的走向只由测试自己决定，不受默认阈值干扰。
_LENIENT = ChannelConfig(min_chars=1)


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


def _el(ref="a.png", alt="示意图"):
    return Element(kind="image", order=0, text=alt, image_ref=ref)


def _run(tmp_path, ocr, vlm, cfg=_LENIENT):
    (tmp_path / "a.png").write_bytes(PNG)
    return resolve_image_element(
        _el(),
        ocr,
        vlm,
        base_dir=tmp_path,
        cache_dir=tmp_path / "c",
        fetcher=lambda u: PNG,
        cfg=cfg,
    )


# ── 规则表逐行 ────────────────────────────────────────────────


def test_row1_ocr_only_when_vlm_unavailable(tmp_path):
    out = _run(tmp_path, FakeOcr("服务状态正常"), None)
    assert "服务状态正常" in out.text
    assert degrade.VLM_UNAVAILABLE in out.degrade


def test_row2_text_screenshot_uses_ocr_only(tmp_path):
    out = _run(tmp_path, FakeOcr("报错：连接超时"), FakeVlm(image_type="截图", content="一个报错弹窗"))
    assert "报错：连接超时" in out.text
    assert "一个报错弹窗" not in out.text  # 纯文字截图不留多模态描述
    assert out.degrade == ()


def test_row3_chart_produces_both_channels(tmp_path):
    out = _run(
        tmp_path,
        FakeOcr("开始 结束"),
        FakeVlm(image_type="流程图", content="三步流程", relations="开始→处理→结束"),
    )
    assert "开始 结束" in out.text  # OCR 通道
    assert "三步流程" in out.text  # 多模态通道（两者兼有则都产出）
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


def test_min_chars_threshold_decides_text_vs_visual(tmp_path):
    """阈值是「文字为主」与「视觉为主」的分界：同一张图，OCR 抽到的字数不同，
    走的分支就不同。用默认阈值（10）跑，不用 _LENIENT。
    """
    vlm = FakeVlm(image_type="照片", content="一张现场照片")
    cfg = ChannelConfig()

    # 只抽到一两个字：不足以断定这是文字图 → 丢掉 OCR，只留多模态。
    # 不记降级码：图里本来就没多少字，这不是缺陷。
    short = _run(tmp_path, FakeOcr("短"), vlm, cfg=cfg)
    assert "一张现场照片" in short.text
    assert "文字:" not in short.text
    assert short.degrade == ()

    # 同样这张图，OCR 抽到足够多的字 → 认定为文字为主，两条通道都产出
    long = _run(tmp_path, FakeOcr("服务状态正常，CPU 占用 87%"), vlm, cfg=cfg)
    assert "文字: 服务状态正常，CPU 占用 87%" in long.text
    assert "一张现场照片" in long.text


# ── 解析失败 / 取图失败 ───────────────────────────────────────


def test_ocr_engine_failure_is_recorded_as_ocr_failed(tmp_path):
    """「引擎抛异常」与「图里没字」必须区分开——前者是缺陷，后者是图的属性。

    注意这里 OCR 挂了但多模态有产出，`ocr_failed` 仍然要记：通道级失败永远记。
    """

    class BoomOcr:
        def extract(self, image):
            raise ValueError("不是图片")

    out = _run(tmp_path, BoomOcr(), FakeVlm(image_type="照片", content="一张现场照片"))
    assert degrade.OCR_FAILED in out.degrade
    assert "一张现场照片" in out.text


def test_vlm_engine_failure_is_recorded_as_vlm_failed(tmp_path):
    """多模态抛异常（超时/限流）也要落码，且不影响 OCR 的产出"""

    class BoomVlm:
        def describe(self, image):
            raise RuntimeError("网关超时")

    out = _run(tmp_path, FakeOcr("服务状态正常"), BoomVlm())
    assert degrade.VLM_FAILED in out.degrade
    assert "服务状态正常" in out.text


def test_bad_vlm_json_keeps_raw_text(tmp_path):
    """解析失败也要把原文留下"""
    vlm = FakeVlm(image_type="其他", content="", raw="```json\n{坏掉的\n```")
    out = _run(tmp_path, FakeOcr(""), vlm)
    assert "坏掉的" in out.text
    assert degrade.VLM_BAD_JSON in out.degrade


def test_fetch_failure_still_yields_element_with_degrade(tmp_path):
    out = resolve_image_element(
        _el(ref="http://x/nope.png"),
        FakeOcr("x"),
        FakeVlm(),
        base_dir=tmp_path,
        cache_dir=tmp_path / "c",
        fetcher=lambda u: (_ for _ in ()).throw(OSError("404")),
    )
    assert degrade.IMAGE_FETCH_FAILED in out.degrade
    assert out.text.strip() != ""


# ── 合并格式与元数据 ──────────────────────────────────────────


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
