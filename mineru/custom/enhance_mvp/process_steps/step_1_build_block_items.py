"""Stage 1: BlockIndexer — 扫描 middle_json['pdf_info'][*]['para_blocks']，输出 TitleItem/BodyItem 扁平序列。

关键约束（与当前落地保持一致）：
- 以 title 作为分割线，产出 [TitleItem, BodyItem, TitleItem, BodyItem, ...]
- refs 采用 layout.pdf 红色编号口径：普通块占 1 个号；视觉父块不占号，子块逐个占号（跳过 cross_page）
- refs 固定为四元组：[page_no, layout_no, type, bbox_str]
- body_text 严格复用 raw_md 渲染（make_blocks_to_markdown，固定 MM_MD）

BlockUnit 设计说明：
- 每个 para_block（无论含多少 lines）对应一个 BlockUnit，是切割的最小原子单元。
- 普通块（text/equation 等）：rendered_text 为整块渲染结果，block_refs 含 1 个外层 bbox ref。
- 视觉父块（table/image/chart/code）：rendered_text 为整块渲染结果，block_refs 含各子块 ref
  （每个子块独占一个 layout_no，跳过 cross_page 子块）。
- Stage 2 对 BodyItem.block_items 做切片即可精确提取子块范围的内容与引用，无需字符偏移对齐。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    make_blocks_to_markdown as _make_blocks_to_markdown,
    merge_para_with_text as _merge_para_text,
)
from mineru.utils.enum_class import BlockType as _BT
from mineru.utils.enum_class import MakeMode as _MakeMode
from mineru.utils.enum_class import SplitFlag as _SplitFlag

# 视觉父块类型（不占 layout_no，由子块各自占号）
_LAYOUT_VISUAL_PARENT_TYPES = frozenset({_BT.IMAGE, _BT.CHART, _BT.CODE, _BT.TABLE})
# 标题类型
_TITLE_TYPES = frozenset({_BT.TITLE, _BT.DOC_TITLE, _BT.PARAGRAPH_TITLE})


def _bbox_str(val: object) -> str:
    """将 bbox 列表转为逗号分隔的四元组字符串，供 ref 使用。"""
    if isinstance(val, list) and len(val) >= 4:
        return ",".join(str(v) for v in val[:4])
    return ""


def _extract_spans_text(sub_block: dict) -> str:
    """从子块所有 lines → spans 提取并拼接 content 文字。"""
    parts: list[str] = []
    for line in sub_block.get("lines") or []:
        for span in line.get("spans") or []:
            content = span.get("content", "").strip()
            if content:
                parts.append(content)
    return " ".join(parts)


def _extract_visual_meta(
    para_block: dict,
    btype: object,
) -> tuple[str, str, str]:
    """从视觉父块的子块列表提取 (img_path, caption_text, footnote_text)。

    决策规则：
    - IMAGE / CHART：img_path 来自 *_body span["image_path"]；
    - TABLE：HTML 非空时 img_path=""（标记为无需增强）；HTML 为空时取 img_path。
    - CODE：img_path 始终为空（代码文本已提取，无需视觉增强）。
    """
    img_path = ""
    caption_text = ""
    footnote_text = ""
    has_html = False

    for sub in para_block.get("blocks") or []:
        if not isinstance(sub, dict):
            continue
        sub_type = str(sub.get("type", ""))

        if sub_type.endswith("_body"):
            # 提取图片路径（image_body / chart_body / table_body fallback）
            for line in sub.get("lines") or []:
                for span in line.get("spans") or []:
                    # 优先取 image_path 字段（MinerU 标准字段名）
                    path = span.get("image_path") or span.get("img_path") or ""
                    if path and not img_path:
                        img_path = str(path)
                    # TABLE：检查是否有 HTML
                    if span.get("html"):
                        has_html = True

        elif sub_type.endswith("_caption"):
            text = _extract_spans_text(sub)
            if text:
                caption_text = text

        elif sub_type.endswith("_footnote"):
            text = _extract_spans_text(sub)
            if text:
                footnote_text = text

    # TABLE with HTML → 不做图片增强（HTML 已语义化）
    if btype == _BT.TABLE and has_html:
        img_path = ""

    # CODE → 不做图片增强（代码文本已提取）
    if btype == _BT.CODE:
        img_path = ""

    return img_path, caption_text, footnote_text


@dataclass
class BlockUnit:
    """单个 para_block 粒度的原子内容单元，将渲染文本与物理引用绑定。

    是 Stage 2 进行子块切割的最小不可分单元：
    - 普通块：1 段 rendered_text + 1 个 ref（外层 bbox）。
    - 视觉父块（TABLE/IMAGE/CHART/CODE）：1 段 rendered_text（整块渲染）
      + N 个 ref（各非 cross_page 子块分别占 1 个 layout_no）。

    Stage 2 通过切片 BodyItem.block_items[a:b] 提取子块范围，
    子块的 block_refs / body_text / char_count 均从子列表聚合得到，天然正确。

    图片增强字段（Stage 1.5 使用，对 Stage 2–7 透明）：
    - img_path: image/chart BlockUnit 的图片文件路径（含 hash/UUID），空字符串表示无需增强。
      TABLE 类型若 HTML 非空则不存（空字符串），确保 Stage 1.5 `if not bu.img_path` 自动跳过。
    - caption_text: 完整图题文字（来自 *_caption 子块），不截断。
    - footnote_text: 图注/补充说明（来自 *_footnote 子块），可为空。
    """

    rendered_text: str       # para_block 整体的 markdown 渲染结果（不含末尾 \n\n）
    block_refs: list[list]   # 四元组 ref 列表：[[page_no, layout_no, type, bbox_str], ...]
    char_count: int          # len(rendered_text)，Stage 2 用于 max_chunk_chars 约束
    image_count: int = 0     # 该块是否为 IMAGE/CHART（0 或 1）
    table_count: int = 0     # 该块是否为 TABLE（0 或 1）
    # ── 图片增强字段（Stage 1.5 专用）──────────────────────────────────────
    img_path: str = ""       # 图片文件路径；空 = 无需增强（TABLE with HTML / CODE / 普通块）
    caption_text: str = ""   # 完整图题（*_caption 子块全文拼接）
    footnote_text: str = ""  # 图注/补充（*_footnote 子块全文拼接）
    image_description: str = ""  # Stage 1.5 成功后写入的多模态增强描述；空 = 未增强


@dataclass
class TitleItem:
    """标题分割点"""
    title_text: str
    block_ref: list  # 标题自身 ref：[page_no, layout_no, type, bbox_str]


@dataclass
class BodyItem:
    """标题之间的一段内容聚合。

    block_items 是核心数据结构，每个元素对应一个 para_block 粒度的 BlockUnit，
    按文档顺序排列，保留了渲染文本与物理引用的精确对应关系。

    Stage 2 切分长 BodyItem 时，对 block_items 做切片即可，每个子块的
    block_refs / body_text / char_count 等均从子列表聚合，天然正确。

    body_text / block_refs / char_count / image_count / table_count
    均为聚合 property，从 block_items 实时推导，对下游保持完全向后兼容。
    """

    block_items: list[BlockUnit] = field(default_factory=list)

    @property
    def body_text(self) -> str:
        """各 BlockUnit 渲染文本拼接，块间以 \\n\\n 分隔（与原 raw_md 渲染口径一致）。"""
        return "".join(bu.rendered_text + "\n\n" for bu in self.block_items if bu.rendered_text)

    @property
    def block_refs(self) -> list[list]:
        """所有 BlockUnit 的 refs 展开为平铺列表。"""
        result: list[list] = []
        for bu in self.block_items:
            result.extend(bu.block_refs)
        return result

    @property
    def char_count(self) -> int:
        return sum(bu.char_count for bu in self.block_items)

    @property
    def image_count(self) -> int:
        return sum(bu.image_count for bu in self.block_items)

    @property
    def table_count(self) -> int:
        return sum(bu.table_count for bu in self.block_items)


def build_block_items(
    middle_json: dict[str, object],
) -> list[TitleItem | BodyItem]:
    """返回文档顺序的 TitleItem/BodyItem 列表。

    每个 BodyItem.block_items 中，BlockUnit 按 para_block 出现顺序排列，
    Stage 2 可通过切片 block_items[a:b] 精确提取任意子块范围的内容与引用。
    """
    pdf_info: list[dict] = middle_json.get("pdf_info", [])
    current_body: BodyItem | None = None
    items: list[TitleItem | BodyItem] = []

    for page_info in pdf_info:
        if not isinstance(page_info, dict):
            continue
        page_no = page_info.get("page_idx", 0) + 1  # 转为 1-based
        para_blocks: list[dict] = page_info.get("para_blocks", []) or []
        layout_no = 0  # per-page 编号计数器（严格复刻 draw_layout_bbox 的 append 顺序）

        for para_block in para_blocks:
            if not isinstance(para_block, dict):
                continue
            btype = para_block.get("type", "")
            bbox = para_block.get("bbox")

            # ── 标题块 ──────────────────────────────────────────────────────
            if btype in _TITLE_TYPES:
                layout_no += 1  # 先占号，与 layout.pdf 红色编号口径一致
                title_text = _merge_para_text(para_block).strip()
                if not title_text:
                    # 标题块内容为空，已占号但不创建 TitleItem
                    continue
                block_ref = [page_no, layout_no, str(btype), _bbox_str(bbox)]
                if current_body is not None and current_body.block_items:
                    items.append(current_body)
                current_body = None

                items.append(TitleItem(title_text=title_text, block_ref=block_ref))
                current_body = BodyItem()

            # ── 视觉父块（TABLE / IMAGE / CHART / CODE）──────────────────────
            elif btype in _LAYOUT_VISUAL_PARENT_TYPES:
                if current_body is None:
                    current_body = BodyItem()

                # 父块本身不占 layout_no；子块各自占号（跳过 cross_page）
                refs: list[list] = []
                for sub_block in para_block.get("blocks", []) or []:
                    if not isinstance(sub_block, dict):
                        continue
                    if sub_block.get(_SplitFlag.CROSS_PAGE, False):
                        continue
                    layout_no += 1
                    refs.append(
                        [
                            page_no,
                            layout_no,
                            str(sub_block.get("type", "unknown")),
                            _bbox_str(sub_block.get("bbox")),
                        ]
                    )

                rendered = _make_blocks_to_markdown([para_block], _MakeMode.MM_MD, img_buket_path="")
                rendered_text = rendered[0] if rendered else ""

                if not refs and not rendered_text:
                    continue

                # 提取图片增强元数据（img_path / caption_text / footnote_text）
                img_path, caption_text, footnote_text = _extract_visual_meta(para_block, btype)

                current_body.block_items.append(
                    BlockUnit(
                        rendered_text=rendered_text,
                        block_refs=refs,
                        char_count=len(rendered_text),
                        image_count=1 if btype in (_BT.IMAGE, _BT.CHART) else 0,
                        table_count=1 if btype == _BT.TABLE else 0,
                        img_path=img_path,
                        caption_text=caption_text,
                        footnote_text=footnote_text,
                    )
                )

            # ── 普通块（text / interline_equation 等）───────────────────────
            else:
                if current_body is None:
                    current_body = BodyItem()

                layout_no += 1  # 先占号，与 layout.pdf 红色编号口径一致

                # lines 为空，或所有 span content 均为空 → 无实际内容，已占号但跳过
                lines = para_block.get("lines") or []
                if not lines or not any(
                    s.get("content", "").strip()
                    for line in lines
                    for s in line.get("spans", [])
                ):
                    continue

                rendered = _make_blocks_to_markdown([para_block], _MakeMode.MM_MD, img_buket_path="")
                rendered_text = rendered[0] if rendered else ""

                if not rendered_text:
                    continue

                current_body.block_items.append(
                    BlockUnit(
                        rendered_text=rendered_text,
                        block_refs=[[page_no, layout_no, str(btype), _bbox_str(bbox)]],
                        char_count=len(rendered_text),
                        image_count=0,
                        table_count=0,
                    )
                )

    # 处理遗留的末尾 body
    if current_body is not None and current_body.block_items:
        items.append(current_body)

    title_count = sum(1 for x in items if isinstance(x, TitleItem))
    body_count = len(items) - title_count
    logger.debug(f"[BlockIndexer] pages={len(pdf_info)} titles={title_count} bodies={body_count}")
    return items
