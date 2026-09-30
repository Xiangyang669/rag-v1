from ragv1.ingest.parser import parse_document

DOC = """---
title: Admin Service
sidebar_position: 0
---
# Admin Service

正文第一段。

## 子节

正文第二段。
"""


def test_parse_strips_frontmatter():
    doc = parse_document(DOC, doc_id="a.md")
    texts = [b.text for b in doc.blocks]
    assert all("sidebar_position" not in t for t in texts)
    assert all("title: Admin Service" not in t for t in texts)


def test_parse_keeps_heading_levels():
    doc = parse_document(DOC, doc_id="a.md")
    headings = [(b.level, b.title) for b in doc.blocks if b.level > 0]
    assert headings == [(1, "Admin Service"), (2, "子节")]


# ── 修复轮：评审发现 2 ────────────────────────────────────────
def test_hash_inside_code_fence_is_not_a_heading():
    """代码块里的 # 是注释，不是标题。

    真实语料（107 篇）里有 6 篇共 18 行这种内容；误判会产生幽灵 section，
    让后续块的 heading_path 全错，图谱也会长出假节点。
    """
    doc = parse_document("# Real Heading\n\n```bash\n# a shell comment\nls -la\n```\n", doc_id="d.md")
    titles = [b.title for b in doc.blocks if b.level > 0]
    assert titles == ["Real Heading"]


def test_code_fence_content_is_preserved_as_body():
    doc = parse_document("# H\n\n```python\nx = 1\n```\n", doc_id="d.md")
    body = "\n".join(b.text for b in doc.blocks)
    assert "x = 1" in body
