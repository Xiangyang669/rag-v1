"""表格的识别、规范化与切分。

表格是唯一一类「切错就等于内容报废」的正文——普通段落切在哪儿都还能读，
表格切在行中间、或后续块丢掉表头，那几行就是废数据。所以切分规则要专门测：
每块必须重复表头；宁可让单行超限也不切断它。
"""

from ragv1.ingest import degrade
from ragv1.ingest.table import normalize_table, split_row, split_table


def test_is_separator_and_table_lines():
    from ragv1.ingest.table import is_separator_line, is_table_line

    assert is_table_line("| a | b |")
    assert is_table_line("  | a | b |")  # 允许前导空白（语料实测存在）
    assert not is_table_line("文本 | 竖线")
    assert is_separator_line("| --- | :-- |")
    assert is_separator_line("|---|")
    assert not is_separator_line("| a | b |")


def test_split_row_handles_escaped_pipe():
    """转义竖线不能多切出一列"""
    assert split_row(r"| a \| b | c |") == ["a | b", "c"]


def test_split_row_only_treats_pipe_and_backslash_as_escape():
    """Markdown 只定义 \\| 与 \\\\ 两种转义。

    其余 \\X 必须保持字面——技术语料里到处是 Windows 路径与正则，
    把它们里的反斜杠吃掉是静默的数据损坏。
    """
    assert split_row(r"| C:\temp | x |") == ["C:\\temp", "x"]
    assert split_row(r"| regex \d+ | y |") == ["regex \\d+", "y"]
    assert split_row(r"| a \\ b | c |") == ["a \\ b", "c"]


def test_normalize_strips_cells_and_keeps_shape():
    assert normalize_table(["| a |b|", "|---|---|", "| 1 | 2 |"]) == (
        "| a | b |\n| --- | --- |\n| 1 | 2 |"
    )


def test_normalize_rejects_malformed_column_count():
    """分隔行列数与表头不一致 → 返回 None 让调用方降级保留原文"""
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
        assert p.splitlines()[0] == "| a | b |"  # 每块都带表头
        assert p.splitlines()[1] == "| --- | --- |"
    assert codes == ()
    assert "\n".join(parts).count("| x") == 20 * 1  # 一行都没丢


def test_split_table_row_longer_than_budget_is_not_cut():
    """宁可超限也不切断行"""
    long_row = "| " + "y" * 500 + " |"
    md = "\n".join(["| a |", "| --- |", long_row])
    parts, codes = split_table(md, budget=50)
    assert any(long_row in p for p in parts)  # 整行完整出现在某一块
    assert degrade.TABLE_ROW_TOO_LONG in codes


def test_split_table_header_only_table_is_one_block():
    """零数据行的空表不能抛异常，也不能产出空块"""
    md = "| a | b |\n| --- | --- |"
    parts, codes = split_table(md, budget=1000)
    assert parts == [md]
    assert codes == ()
