"""凭据解析的测试。

回归背景：早先的实现是「从当前目录向上找 .env」，但 `practice/` 不是
`rag_v1/` 的祖先——生产路径永远找不到 key。这里把它钉住。
"""

import pytest

from ragv1 import embedding
from ragv1.embedding import load_key, resolve_api_key

KEY = "SILICONFLOW_API_KEY"


def test_env_file_candidates_cover_both_layouts():
    """两种布局都要能跑：独立仓库用项目自己的 .env，留在 ai-agent-90days
    里时用上层 practice/.env。就近优先。"""
    candidates = embedding.ENV_FILE_CANDIDATES
    assert candidates[0] == embedding._PROJECT_DIR / ".env"
    assert candidates[1].parts[-2:] == ("practice", ".env")
    assert all(c.is_absolute() for c in candidates)


def test_resolve_uses_first_existing_candidate(monkeypatch, tmp_path):
    monkeypatch.delenv(KEY, raising=False)
    first = tmp_path / "a.env"
    first.write_text(f"{KEY}=first\n", encoding="utf-8")
    second = tmp_path / "b.env"
    second.write_text(f"{KEY}=second\n", encoding="utf-8")
    monkeypatch.setattr(embedding, "ENV_FILE_CANDIDATES", (first, second))
    assert resolve_api_key() == "first"


def test_resolve_skips_missing_candidate(monkeypatch, tmp_path):
    monkeypatch.delenv(KEY, raising=False)
    missing = tmp_path / "missing.env"
    present = tmp_path / "b.env"
    present.write_text(f"{KEY}=second\n", encoding="utf-8")
    monkeypatch.setattr(embedding, "ENV_FILE_CANDIDATES", (missing, present))
    assert resolve_api_key() == "second"


def test_load_key_reads_value(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"{KEY}=abc123\nother=xxx\n", encoding="utf-8")
    assert load_key(KEY, env) == "abc123"


def test_resolve_prefers_env_var(monkeypatch):
    monkeypatch.setenv(KEY, "from-env-var")
    assert resolve_api_key() == "from-env-var"


def test_resolve_falls_back_to_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv(KEY, raising=False)
    env = tmp_path / ".env"
    env.write_text(f"{KEY}=from-file\n", encoding="utf-8")
    assert resolve_api_key(env_file=env) == "from-file"


def test_missing_key_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv(KEY, raising=False)
    with pytest.raises(RuntimeError, match=KEY):
        resolve_api_key(env_file=tmp_path / "does-not-exist.env")
