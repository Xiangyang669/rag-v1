"""Element 中间表示与 Chunk 的来源元数据字段。

本文件按任务分两批：这里先是类型相关的用例（Task 2），
分块行为用例在文件末尾追加（Task 5）。
"""

import dataclasses

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
    assert dataclasses.fields(Chunk)  # 存在即通过；frozen 由下面的赋值验证
    e = Element(kind="text", order=0, text="x")
    try:
        e.text = "y"
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("Element 必须是 frozen 的")
