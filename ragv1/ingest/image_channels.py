"""图片双通道：OCR + 多模态，按内容分流，最后合并。

一条图片元素要回答两个不同的问题：

- **图里写了什么字**（扫描件、报错截图、表格截图）→ OCR 通道
- **这张图画的是什么**（流程图、架构图、照片）→ 多模态通道

两条通道**互补**，不是二选一：一张流程图往往既有节点上的文字（OCR 拿得到），
又有那层文字表达不出的关系（多模态拿得到）。所以「两者兼有」时要**都产出**。

分流规则写成一张显式的表（见 resolve_image_element 的说明），而不是散落的
阈值判断——这样每行分支都能被测试单独钉住。
"""

from dataclasses import dataclass

from ragv1 import config


@dataclass(frozen=True)
class ChannelConfig:
    """图片通道的可调项。默认值取自 config，测试里可整包替换。"""

    min_chars: int = config.OCR_MIN_CHARS
    min_conf: float = config.OCR_MIN_CONF
    enabled: bool = config.ENABLE_IMAGE
