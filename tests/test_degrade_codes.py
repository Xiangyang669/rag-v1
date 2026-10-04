"""降级原因码表。

入库链路上任何解析失败都必须落一个码，不允许静默丢弃。码表集中在一处，
是为了让「产出的码」可枚举、可断言——散落的字符串字面量做不到这一点。
"""

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


def test_all_codes_count():
    """数量写死是为了让「不小心删了一个码」或「加了码没进 ALL_CODES」都被发现。"""
    assert len(degrade.ALL_CODES) == 15
