from ragv1.ingest.parser import parse_document
from ragv1.ingest.chunker import chunk_document, over_cap_ratio
from ragv1.config import MAX_CHARS


def test_chunk_heading_path():
    doc = parse_document("# A\n\nx\n\n## B\n\ny\n", doc_id="d.md")
    chunks = chunk_document(doc, max_chars=MAX_CHARS)
    paths = [c.heading_path for c in chunks]
    assert ("A",) in paths
    assert ("A", "B") in paths


def test_chunk_ids_are_deterministic():
    doc = parse_document("# A\n\nx\n", doc_id="d.md")
    assert [c.chunk_id for c in chunk_document(doc, MAX_CHARS)] == [
        c.chunk_id for c in chunk_document(doc, MAX_CHARS)
    ]


def test_doc_without_headings_still_chunks():
    doc = parse_document("只有正文，没有任何标题。", doc_id="d.md")
    chunks = chunk_document(doc, max_chars=MAX_CHARS)
    assert len(chunks) >= 1


def test_overlong_section_is_split():
    body = "。" * 5000
    doc = parse_document(f"# A\n\n{body}\n", doc_id="d.md")
    chunks = chunk_document(doc, max_chars=1200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1200 for c in chunks)


def test_split_does_not_cross_headings():
    doc = parse_document("# A\n\n" + "。" * 3000 + "\n\n# B\n\n短\n", doc_id="d.md")
    chunks = chunk_document(doc, max_chars=1200)
    assert all(c.heading_path in (("A",), ("B",)) for c in chunks)


def test_no_empty_chunks():
    doc = parse_document("# A\n\n" + "。" * 3000 + "\n", doc_id="d.md")
    assert all(c.text.strip() for c in chunk_document(doc, max_chars=1200))


def test_over_cap_ratio_exposed():
    doc = parse_document("# A\n\n" + "。" * 3000 + "\n", doc_id="d.md")
    assert over_cap_ratio(doc, max_chars=1200) > 0.0
