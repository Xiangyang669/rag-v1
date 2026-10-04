"""图片取回：远程下载 / 本地路径 + 本地缓存。

语料里的图片引用大多是**远程 URL**（实测 184 处全在 raw.githubusercontent.com），
所以想 OCR 就得先有像素——这一步是双通道的前置。

两条硬规则：

1. **失败绝不向上抛**。上百处引用里必然有坏链接，一个 404 不能毁掉整篇文档的
   入库——所有失败都转成降级码返回。
2. **缓存是优化，不是依赖**。缓存目录不可写时照样把字节交出去；命中缓存时
   不再发起网络请求（否则重跑一次入库就是几百次下载）。
"""

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ragv1.ingest import degrade

# 远程下载超时（秒）。卡住的下载不该拖垮整次入库。
FETCH_TIMEOUT = 10


@dataclass(frozen=True)
class ImageBytes:
    """取回结果。data 为 None 表示失败，此时 degrade 里必有原因码。"""

    data: bytes | None
    path: str | None = None
    degrade: tuple[str, ...] = ()


def _default_fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT) as resp:  # noqa: S310
        return resp.read()


def _cache_path(ref: str, cache_dir: Path) -> Path:
    """按 ref 的哈希命名，保留原后缀（有些引擎靠后缀判断格式）。"""
    digest = hashlib.sha1(ref.encode("utf-8")).hexdigest()[:16]
    suffix = Path(urllib.request.urlparse(ref).path).suffix or ".png"
    return cache_dir / f"{digest}{suffix}"


def _is_remote(ref: str) -> bool:
    return ref.startswith("http://") or ref.startswith("https://")


def resolve_image_bytes(
    ref: str,
    base_dir: Path,
    cache_dir: Path,
    fetcher: Callable[[str], bytes] | None = None,
) -> ImageBytes:
    """取回图片字节。任何失败都返回 ImageBytes(None, ..., degrade=(原因,))。"""
    if not _is_remote(ref):
        path = (base_dir / ref) if not Path(ref).is_absolute() else Path(ref)
        if not path.exists():
            return ImageBytes(None, None, (degrade.IMAGE_MISSING,))
        try:
            return ImageBytes(path.read_bytes(), str(path))
        except OSError:
            return ImageBytes(None, str(path), (degrade.IMAGE_MISSING,))

    cached = _cache_path(ref, cache_dir)
    if cached.exists():
        try:
            return ImageBytes(cached.read_bytes(), str(cached))
        except OSError:
            pass  # 缓存读坏了就当没缓存，走下载

    fetch = fetcher or _default_fetch
    try:
        data = fetch(ref)
    except Exception:  # noqa: BLE001 —— 降级点：任何网络/解析异常都不许上抛
        return ImageBytes(None, None, (degrade.IMAGE_FETCH_FAILED,))

    # 缓存写失败只影响下次，不影响这次
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    except OSError:
        return ImageBytes(data, None)

    return ImageBytes(data, str(cached))
