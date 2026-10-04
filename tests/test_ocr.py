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


def test_extract_on_engine_exception_returns_empty_result(monkeypatch):
    """拿到非图片字节时不能抛"""
    eng = build_ocr_engine("rapidocr")

    class BoomImpl:
        def __call__(self, img):
            raise ValueError("不是图片")

    monkeypatch.setattr(eng, "_impl", BoomImpl(), raising=False)
    r = eng.extract(b"<html>404</html>")
    assert r.text == "" and r.confidence == 0.0
