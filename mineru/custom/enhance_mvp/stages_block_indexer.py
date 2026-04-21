"""Stage 1: BlockIndexer — 替换 SectionBuildStage。

从 content_list_v2 中扫描文档物理结构，输出两种独立数据结构：
  TitleItem  — 每个 title block 一个
  BodyItem   — 紧跟某 TitleItem（或文档前置）的非 title 内容聚合体

设计原则：
  - 数组顺序即文档顺序，无需 seq_id / after_seq_id 存储关系
  - block_refs 为三元列表 [page_no, block_no, type]，页码与块号均为 1-based
    （与 content_list_v2 的外层 page_idx + 内层 block_idx 枚举对应，+1 对齐）
  - 空 BodyItem（block_refs=[] 且 char_count=0）不发射，消除噪音
  - 多个连续 title → 连续 TitleItem，中间无占位 BodyItem
  - build_skip_set() 动态合并内置跳过类型 + custom_config.discard.types 映射
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from mineru.custom.custom_config_loader import CustomConfig

# ---------------------------------------------------------------------------
# 常量：始终跳过的 ContentTypeV2 类型（不进 block_refs，不进文本）
# ---------------------------------------------------------------------------

_ALWAYS_SKIP_V2: frozenset[str] = frozenset({
    "page_header",
    "page_footer",
    "page_number",
    "page_aside_text",
    "page_footnote",
})

# BlockType（middle_json） → ContentTypeV2（content_list_v2）映射
_BLOCK_TYPE_TO_CONTENT_TYPE_V2: dict[str, set[str]] = {
    "header":               {"page_header"},
    "footer":               {"page_footer"},
    "page_number":          {"page_number"},
    "aside_text":           {"page_aside_text"},
    "page_footnote":        {"page_footnote"},
    "image":                {"image"},
    "table":                {"simple_table", "complex_table"},
    "chart":                {"chart"},
    "interline_equation":   {"equation_interline"},
    "seal":                 {"seal"},
    "code":                 {"code", "algorithm"},
    "list":                 {"list"},
    "index":                {"index"},
    "ref_text":             {"list"},
    "text":                 {"paragraph"},
    "abstract":             {"paragraph"},
    "title":                {"title"},
}

# 文本类（进 block_refs + 进文本）
_TEXT_TYPES_V2: frozenset[str] = frozenset({"paragraph", "list", "index"})

# 视觉类（进 block_refs，不进文本，可选 caption）
_VISUAL_TYPES_V2: frozenset[str] = frozenset({
    "image", "simple_table", "complex_table", "chart",
    "equation_interline", "seal", "code", "algorithm",
})


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class TitleItem:
    """content_list_v2 中的 title block 的结构化表示。"""
    title_text: str
    block_ref: list   # [page_no_1based, block_no_1based, "title"]


@dataclass
class BodyItem:
    """紧跟某 TitleItem（或文档前置）的非 title 内容聚合体。"""
    char_count: int = 0
    preview: str = ""                                   # finalize_body 后填写
    block_refs: list[list] = field(default_factory=list)  # [[page_no, block_no, type], ...]
    body_text: str = ""                                 # 完整正文文本（供 LLM 使用）


# ---------------------------------------------------------------------------
# Skip set 构建
# ---------------------------------------------------------------------------

def build_skip_set(custom_config: CustomConfig) -> frozenset[str]:
    """动态构建 skip set：内置跳过类型 ∪ custom_config.discard.types 映射。

    custom_config.discard.types 中存储的是 BlockType 字符串（middle_json 层的类型），
    需要映射到 ContentTypeV2 才能与 content_list_v2 的 block["type"] 比较。
    """
    skip: set[str] = set(_ALWAYS_SKIP_V2)
    discard_cfg = custom_config.discard or {}
    custom_types: list[str] = discard_cfg.get("types") or []
    for block_type in custom_types:
        mapped = _BLOCK_TYPE_TO_CONTENT_TYPE_V2.get(block_type)
        if mapped:
            skip.update(mapped)
        # 未知 block_type 静默忽略（validate_custom_config 已做校验）
    return frozenset(skip)


# ---------------------------------------------------------------------------
# 文本提取工具
# ---------------------------------------------------------------------------

def _extract_inline_list(items: list[Any]) -> str:
    """从 {type, content} 列表中拼接文本（兼容 title_content / paragraph_content / spans）。"""
    parts: list[str] = []
    for span in items:
        if not isinstance(span, dict):
            continue
        if span.get("type") in {"text", "equation_inline"}:
            c = span.get("content", "")
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
    return " ".join(parts)


def _extract_spans_text(spans: list[Any]) -> str:
    """从 spans 列表中提取纯文本（兼容旧 spans 格式，供测试夹具使用）。"""
    return _extract_inline_list(spans)


def _extract_title_text(block: dict[str, Any]) -> str:
    """从 title block 提取标题文本。

    兼容两种格式：
      - 实际 content_list_v2：content.title_content（数组）
      - 测试夹具：content.spans（数组）
    """
    content = block.get("content") or {}
    if not isinstance(content, dict):
        return ""
    # 优先：真实格式 title_content
    title_items = content.get("title_content")
    if isinstance(title_items, list):
        text = _extract_inline_list(title_items)
        if text:
            return text
    # fallback：spans（测试夹具格式）
    spans = content.get("spans") or []
    text = _extract_inline_list(spans)
    if text:
        return text
    # fallback：直接读 title 字符串字段
    return str(content.get("title", "") or "").strip()


def _extract_text_recursive(block: dict[str, Any]) -> str:
    """递归提取 block 的可读文本（支持 paragraph/list/index 及其 list_items）。

    兼容两种格式：
      - 实际 content_list_v2：paragraph_content / list_items
      - 测试夹具：spans
    """
    btype = block.get("type", "")
    content = block.get("content") or {}
    if not isinstance(content, dict):
        return ""

    if btype == "paragraph":
        # 优先：真实格式 paragraph_content
        para_items = content.get("paragraph_content")
        if isinstance(para_items, list):
            return _extract_inline_list(para_items)
        # fallback：spans（测试夹具格式）
        return _extract_inline_list(content.get("spans") or [])

    if btype in {"list", "index"}:
        items: list[Any] = content.get("list_items") or []
        parts: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_content_list: list[Any] = item.get("item_content") or []
            for sub in item_content_list:
                if not isinstance(sub, dict):
                    continue
                if sub.get("type") in {"text", "equation_inline"}:
                    c = sub.get("content", "")
                    if isinstance(c, str) and c.strip():
                        parts.append(c.strip())
        return "\n".join(parts)

    # 其他文本类型：尝试常见 content 键
    for key in ("paragraph_content", "title_content", "spans"):
        items_list = content.get(key)
        if isinstance(items_list, list) and items_list:
            text = _extract_inline_list(items_list)
            if text:
                return text
    return ""


def _extract_caption_text(block: dict[str, Any]) -> str:
    """从 image/table block 提取 caption 文本（可选）。"""
    content = block.get("content") or {}
    if not isinstance(content, dict):
        return ""
    # image caption
    caption = content.get("image_caption") or content.get("table_caption") or ""
    if isinstance(caption, str):
        return caption.strip()
    if isinstance(caption, list):
        parts: list[str] = []
        for item in caption:
            if isinstance(item, dict):
                spans = item.get("spans") or []
                t = _extract_spans_text(spans)
                if t:
                    parts.append(t)
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts).strip()
    return ""


# ---------------------------------------------------------------------------
# Preview 截取
# ---------------------------------------------------------------------------

def _finalize_body(body: BodyItem) -> BodyItem:
    """计算 BodyItem.preview（首尾采样）。

    策略（基于 body_text 去首尾空白后的长度）：
      =0       → ""
      ≤50      → 全文
      ≤200     → 前 50 + … + 后 50
      >200     → 前 80 + … + 后 80
    """
    text = body.body_text.strip()
    if not text:
        body.preview = ""
    elif len(text) <= 50:
        body.preview = text
    elif len(text) <= 200:
        body.preview = text[:50] + "…" + text[-50:]
    else:
        body.preview = text[:80] + "…" + text[-80:]
    return body


def _emit_body_if_nonempty(body: BodyItem | None, items: list) -> None:
    """若 body 有内容（有 block_refs 或有字符），则 finalize 后追加到 items。"""
    if body is not None and (body.block_refs or body.char_count > 0):
        items.append(_finalize_body(body))


# ---------------------------------------------------------------------------
# 主扫描算法
# ---------------------------------------------------------------------------

def run_block_indexer(
    content_list_v2: list[list[dict[str, Any]]],
    custom_config: CustomConfig,
) -> list[TitleItem | BodyItem]:
    """Block Indexer 主入口。

    参数:
        content_list_v2: list[list[dict]]，外层按页，内层按 block_idx 顺序
        custom_config:   CustomConfig 实例，用于 build_skip_set

    返回:
        文档物理顺序的 TitleItem / BodyItem 混合列表：
        - 数组顺序即文档顺序，消费侧无需 seq_id / after_seq_id 字段
        - 连续 title 块 → 连续 TitleItem（中间无空 BodyItem）
        - 空 BodyItem（无 block_refs 且 char_count=0）不发射
        - block_refs 为三元列表 [page_no_1based, block_no_1based, type_str]
    """
    skip_types = build_skip_set(custom_config)

    current_body: BodyItem | None = None
    items: list[TitleItem | BodyItem] = []

    for page_idx, page_blocks in enumerate(content_list_v2):
        if not isinstance(page_blocks, list):
            continue
        for block_idx, block in enumerate(page_blocks):
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")

            # ① 丢弃类：完全跳过
            if block_type in skip_types:
                continue

            # 三元引用：1-based 页码 + 块号
            block_ref = [page_idx + 1, block_idx + 1, block_type]

            # ② Title block → 封存当前 body（若非空），创建新 TitleItem
            if block_type == "title":
                _emit_body_if_nonempty(current_body, items)
                current_body = None

                title_text = _extract_title_text(block)
                items.append(TitleItem(title_text=title_text, block_ref=block_ref))

                # 为此 title 准备接收后续正文的 BodyItem（可能为空，最终不发射）
                current_body = BodyItem()

            # ③ 文本类（paragraph / list / index）→ 进 refs + 进文本
            elif block_type in _TEXT_TYPES_V2:
                if current_body is None:
                    # preamble：首个 title 前的正文
                    current_body = BodyItem()
                current_body.block_refs.append(block_ref)
                text = _extract_text_recursive(block)
                if text:
                    current_body.body_text += text + "\n"
                    current_body.char_count += len(text)

            # ④ 视觉类（image/table/chart/equation/seal/code）→ 仅进 refs，不进文本
            elif block_type in _VISUAL_TYPES_V2:
                if current_body is None:
                    current_body = BodyItem()
                current_body.block_refs.append(block_ref)
                # 可选：caption 文本加入 body_text，用标记区分
                caption = _extract_caption_text(block)
                if caption:
                    current_body.body_text += f"[图/表注: {caption}]\n"

            else:
                # 未知类型：仅进 refs（防御性保留）
                if current_body is None:
                    current_body = BodyItem()
                current_body.block_refs.append(block_ref)

    # 封存最后一个 body（若非空）
    _emit_body_if_nonempty(current_body, items)

    title_count = sum(1 for x in items if isinstance(x, TitleItem))
    body_count = sum(1 for x in items if isinstance(x, BodyItem))
    logger.debug(
        f"[BlockIndexer] scanned pages={len(content_list_v2)} "
        f"titles={title_count} bodies={body_count} "
        f"skip_types={sorted(skip_types)}"
    )
    return items
