"""语料入库的端到端：含表格的文档走完 loader → elements → 三路 store。

这是阶段一唯一一条「全链路」测试——单测证明了每一段的行为，这里证明它们
接起来也成立：表格真的以 kind="table" 落到了回源表里，大表真的被切了，
每一块真的都带着表头。
"""

from pathlib import Path

from conftest import fake_embed

from ragv1.ingest import table
from ragv1.ingest.build import build_corpus
from ragv1.store.fts_store import FtsStore

FIXTURES = Path(__file__).parent / "fixtures"


def _build(tmp_path, name="idx"):
    return build_corpus(FIXTURES, tmp_path / name, embed_fn=fake_embed)


def test_build_indexes_table_doc_end_to_end(tmp_path):
    assert _build(tmp_path) > 0
    store = FtsStore(tmp_path / "idx" / "kb.db")

    table_ids = sorted(
        cid for cid in store.chunk_ids() if store.meta_of(cid)["kind"] == "table"
    )
    assert table_ids, "含表文档没有产出任何 table 类型的块"

    parts = [store.meta_of(cid)["part"] for cid in table_ids]
    assert any(p > 0 for p in parts), "大表没有被切分"

    for cid in table_ids:
        lines = store.text_of(cid).splitlines()
        # 块文本 = （可选的 heading 前缀）+ 表头行 + 分隔行 + 数据行。
        # 前缀是可选的——Markdown 的表在标题下才有，PDF 出来的表就没有。
        # 列数也因表而异，所以按「是不是表行/分隔行」判断，不比对具体文字。
        sep_at = next(
            (i for i, ln in enumerate(lines) if table.is_separator_line(ln)), None
        )
        assert sep_at is not None and sep_at >= 1, "块里没有分隔行——表头没跟着切块"
        assert table.is_table_line(lines[sep_at - 1]), "分隔行前面不是表头行"


def test_build_is_deterministic(tmp_path):
    a = _build(tmp_path, "a")
    b = _build(tmp_path, "b")
    assert a == b
    assert FtsStore(tmp_path / "a" / "kb.db").chunk_ids() == FtsStore(
        tmp_path / "b" / "kb.db"
    ).chunk_ids()


def test_build_skips_unknown_suffixes(tmp_path):
    """非 md / pdf 的文件不应让入库崩溃，也不该产出块"""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# H\n\n正文。\n", encoding="utf-8")
    (corpus / "b.json").write_text('{"x": 1}', encoding="utf-8")

    assert build_corpus(corpus, tmp_path / "idx", embed_fn=fake_embed) == 1


def test_build_corpus_runs_image_channels_when_engines_given(tmp_path):
    """给了引擎就得真的用上——否则「从 build_corpus 进来图片永不处理」，
    而每个模块自己的测试还是绿的。这类集成缺口要防。
    """
    from ragv1.ingest.ocr import OcrResult

    class _Ocr:
        def extract(self, image):
            return OcrResult("图里的字：服务状态正常", 0.9, ())

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.png").write_bytes(b"\x89PNG\r\n\x1a\nbody")
    (corpus / "a.md").write_text("# H\n\n![示意图](a.png)\n", encoding="utf-8")

    build_corpus(corpus, tmp_path / "idx", embed_fn=fake_embed, ocr=_Ocr())

    store = FtsStore(tmp_path / "idx" / "kb.db")
    image_ids = [c for c in store.chunk_ids() if store.meta_of(c)["kind"] == "image"]
    assert image_ids, "给了 OCR 引擎却没有产出图片块"
    assert "服务状态正常" in store.text_of(image_ids[0])
    assert store.meta_of(image_ids[0])["image_ref"] == "a.png"
