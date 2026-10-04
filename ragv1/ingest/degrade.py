"""入库侧的降级原因码。

入库链路上任何一步失败——图片下不下来、OCR 抽不出字、表格解析不出结构、
PDF 打不开——都**必须落一个码**，绝不允许静默丢弃。原因码集中在这里，
是为了让「产出的码」可枚举、可断言（tests/test_degrade_codes.py 会校验
产出 ⊆ ALL_CODES），而不是散落成一堆字符串字面量。

分区只是便于阅读，语义上没有层级：所有码都是平级的字符串。
"""

# ── 表格解析 ──
TABLE_UNSTRUCTURED = "table_unstructured"
"""表格不是合法 pipe 表（没有分隔行、或列数不一致）→ 保留原文，标记未结构化。"""

TABLE_ROW_TOO_LONG = "table_row_too_long"
"""单个数据行本身就超过块预算 → 该行独立成块，宁可超限也不切断行。"""

# ── 图片取回 ──
IMAGE_REF_UNRESOLVED = "image_ref_unresolved"
"""引用式图片 ![alt][id] 需要查文末定义表，本期不解析 → 保留原文。"""

IMAGE_FETCH_FAILED = "image_fetch_failed"
"""远程图片下载失败（404 / 超时 / 网络不可达）。"""

IMAGE_MISSING = "image_missing"
"""本地图片路径不存在。"""

# ── OCR 通道 ──
OCR_FAILED = "ocr_failed"
"""OCR 引擎抛异常（多为拿到非图片字节）。"""

OCR_EMPTY = "ocr_empty"
"""OCR 跑通了但没抽到任何文字。"""

OCR_LOW_CONF = "ocr_low_conf"
"""抽到文字但置信度低于阈值，不足以当作「有料」。"""

# ── 多模态通道 ──
VLM_UNAVAILABLE = "vlm_unavailable"
"""未配置 API key、未启用、或引擎名为未知值 → 该通道不参与。"""

VLM_FAILED = "vlm_failed"
"""多模态调用抛异常（超时 / 限流 / 服务端错误）。"""

VLM_BAD_JSON = "vlm_bad_json"
"""多模态返回的不是合法 JSON → 原始返回文本当描述保留，不丢。"""

# ── PDF 版面分析 ──
PDF_OPEN_FAILED = "pdf_open_failed"
"""整份 PDF 打不开（损坏 / 加密 / 根本不是 PDF）→ 产出占位元素，不静默跳过。"""

PDF_PAGE_NO_TEXT_LAYER = "pdf_page_no_text_layer"
"""该页没有文字层（扫描件）→ 整页渲染后走图片双通道。"""

PDF_RENDER_FAILED = "pdf_render_failed"
"""页面渲染成位图失败，或图片区域取不到位图字节。"""

PDF_TABLE_BBOX_MISSING = "pdf_table_bbox_missing"
"""版面分析框出了表格区域，但取不到它的行列内容。"""

# 显式列出，不用 vars() 自省——测试要拿这个集合去校验，自省会让测试变成同义反复。
ALL_CODES: frozenset[str] = frozenset(
    {
        TABLE_UNSTRUCTURED,
        TABLE_ROW_TOO_LONG,
        IMAGE_REF_UNRESOLVED,
        IMAGE_FETCH_FAILED,
        IMAGE_MISSING,
        OCR_FAILED,
        OCR_EMPTY,
        OCR_LOW_CONF,
        VLM_UNAVAILABLE,
        VLM_FAILED,
        VLM_BAD_JSON,
        PDF_OPEN_FAILED,
        PDF_PAGE_NO_TEXT_LAYER,
        PDF_RENDER_FAILED,
        PDF_TABLE_BBOX_MISSING,
    }
)


def normalize(codes) -> tuple[str, ...]:
    """降级码的有序去重。

    排序是硬要求：同一个输入两次运行必须给出逐字节相同的 chunk.meta，
    否则评估数字和跨重启的回源都会漂。
    """
    return tuple(sorted(set(codes)))
