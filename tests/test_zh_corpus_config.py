"""V2 中文主语料的路径常量。

中文语料与英文基准语料必须落在**两套互不覆盖**的索引上：V1 的回归基线
依赖 `.indexes/kb`，V2 建中文索引时若复用它，等于把基线冲掉。
"""

from pathlib import Path

from ragv1 import config


def test_zh_index_dir_is_separate_from_en_index():
    assert config.ZH_INDEX_DIR != config.INDEX_DIR / "kb"
    assert config.ZH_INDEX_DIR.name == "kb_zh"


def test_zh_corpus_dir_is_a_path():
    assert isinstance(config.ZH_CORPUS_DIR, Path)
