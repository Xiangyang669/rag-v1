"""OCR 引擎：惰性加载、输出归一化、失败不抛。

引擎自己**不管降级码**——抽不到字就返回空结果，由调用方按「文本为空/置信度低」
判定该落哪个码。这样引擎可以独立替换，规则只在一处。
"""

from ragv1.ingest.ocr import OcrResult, build_ocr_engine


def test_build_returns_none_for_none_engine():
    assert build_ocr_engine("none") is None


def test_build_returns_none_for_unknown_engine():
    """配错的引擎名不该让入库崩掉，而是干净地降级"""
    assert build_ocr_engine("no-such-engine") is None


def test_rapidocr_engine_is_lazy_until_first_call():
    """import 期不许加载模型"""
    eng = build_ocr_engine("rapidocr")
    assert eng is not None
    assert getattr(eng, "_impl", None) is None  # 还没实例化 rapidocr


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
    assert abs(r.confidence - 0.94) < 1e-6  # 逐行均值


def test_engine_exception_propagates_to_caller(monkeypatch):
    """引擎**不做降级**，异常要冒到调用方。

    本模块的契约是「只管抽字，不管降级码」——降级策略属于 image_channels。
    在这里把异常吞成空结果，会让「引擎抛异常」与「图里没字」变得无法区分，
    `ocr_failed` 也就永远打不出来。
    """
    import pytest

    eng = build_ocr_engine("rapidocr")

    class BoomImpl:
        def __call__(self, img):
            raise ValueError("不是图片")

    monkeypatch.setattr(eng, "_impl", BoomImpl(), raising=False)
    with pytest.raises(ValueError):
        eng.extract(b"<html>404</html>")
