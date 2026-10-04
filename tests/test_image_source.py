"""图片取回：远程下载 / 本地路径 / 缓存。

语料里有上百处图片引用，一个坏链接不能毁掉整篇文档的入库——所有失败
路径都必须转成降级码返回，绝不向上抛。
"""

from pathlib import Path

from ragv1.ingest import degrade
from ragv1.ingest.image_source import resolve_image_bytes

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"


def test_remote_url_is_downloaded_and_cached(tmp_path):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return PNG

    out = resolve_image_bytes(
        "http://x/y.png", tmp_path, tmp_path / "cache", fetcher=fake_fetch
    )
    assert out.data == PNG
    assert out.degrade == ()
    assert Path(out.path).exists() and Path(out.path).read_bytes() == PNG

    # 第二次命中缓存，不再下载
    out2 = resolve_image_bytes(
        "http://x/y.png", tmp_path, tmp_path / "cache", fetcher=fake_fetch
    )
    assert out2.data == PNG
    assert len(calls) == 1, "缓存未命中，重复下载了"


def test_local_relative_path_resolves_against_base_dir(tmp_path):
    (tmp_path / "a.png").write_bytes(PNG)
    out = resolve_image_bytes("a.png", tmp_path, tmp_path / "cache")
    assert out.data == PNG and out.degrade == ()


def test_missing_local_file_degrades(tmp_path):
    """坏引用不能中断入库"""
    out = resolve_image_bytes("nope.png", tmp_path, tmp_path / "cache")
    assert out.data is None
    assert degrade.IMAGE_MISSING in out.degrade


def test_fetch_failure_degrades_and_does_not_raise(tmp_path):
    """URL 404 / 超时"""

    def boom(url):
        raise OSError("404")

    out = resolve_image_bytes(
        "http://x/y.png", tmp_path, tmp_path / "cache", fetcher=boom
    )
    assert out.data is None
    assert degrade.IMAGE_FETCH_FAILED in out.degrade


def test_cache_write_failure_still_returns_bytes(tmp_path):
    """缓存目录不可写时，数据仍要能用——缓存是优化不是依赖"""

    def fake_fetch(url):
        return PNG

    blocked = tmp_path / "blocked"
    blocked.write_text("我是文件不是目录", encoding="utf-8")

    out = resolve_image_bytes(
        "http://x/y.png", tmp_path, blocked, fetcher=fake_fetch
    )
    assert out.data == PNG
