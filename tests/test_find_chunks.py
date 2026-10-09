import importlib.util
from pathlib import Path


def _load():
    spec = importlib.util.spec_from_file_location(
        "find_chunks", Path(__file__).resolve().parents[1] / "scripts" / "find_chunks.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_render_line_contains_id_and_snippet():
    mod = _load()
    line = mod.render_row("cid1", {"doc_id": "d", "page": 3}, "很长的正文" * 40)
    assert "cid1" in line and "d" in line and "3" in line
    assert len(line) < 400  # 摘要必须被截断


def test_render_row_tolerates_missing_meta():
    mod = _load()
    assert "cid1" in mod.render_row("cid1", None, None)
