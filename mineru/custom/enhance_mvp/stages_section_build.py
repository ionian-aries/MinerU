"""Stage 2: SectionBuildStage — 基于 content_list_v2 构建层级化 section tree。

将 content_list_v2 的扁平 block 序列转换为 SectionNode 树（纯规则，无 token/LLM）。

content_list_v2 的真实结构:
    list[list[dict]]  — 外层索引 = page_idx, 内层每个 dict 是一个 block。

每个 block 的通用结构:
    {
        "type": "title" | "paragraph" | "table" | "image" | ...,
        "content": {
            "title_content": [{"type":"text","content":"..."},...],  # title 块
            "paragraph_content": [...],                              # paragraph 块
            "level": 1,                                              # title 块特有
            ...
        },
        "bbox": [x0, y0, x1, y1],  # 可选
    }
"""
import re
from dataclasses import dataclass
from typing import Any, Optional

from mineru.custom.enhance_mvp.schema import SectionNode


# ---------------------------------------------------------------------------
# content_list_v2 结构解析
# ---------------------------------------------------------------------------

# 会产出文本的 content 子键
_TEXT_CONTENT_KEYS = (
    "title_content",
    "paragraph_content",
    "page_header_content",
    "page_footer_content",
    "page_aside_text_content",
    "page_footnote_content",
    "page_number_content",
)

# 需要跳过的非文本 block 类型（不参与 section 切分）
_SKIP_BLOCK_TYPES = frozenset({
    "image", "table", "chart", "seal",
    "equation_interline", "code", "algorithm",
    "page_header", "page_footer", "page_number",
    "page_aside_text", "page_footnote",
})


def _extract_block_text(block: dict[str, Any]) -> str:
    """从一个 content_list_v2 block 中提取可读纯文本。"""
    content_obj = block.get("content")
    if not isinstance(content_obj, dict):
        return ""

    for key in _TEXT_CONTENT_KEYS:
        spans = content_obj.get(key)
        if not isinstance(spans, list):
            continue
        parts: list[str] = []
        for span in spans:
            if isinstance(span, dict):
                c = span.get("content", "")
                if isinstance(c, str) and c.strip():
                    parts.append(c.strip())
        if parts:
            return " ".join(parts)

    return ""


def _is_title_block(block: dict[str, Any]) -> bool:
    """判断 block 是否为标题块。"""
    block_type = str(block.get("type", "")).lower()
    if block_type == "title":
        return True
    content_obj = block.get("content")
    if isinstance(content_obj, dict):
        if content_obj.get("level") and content_obj.get("title_content"):
            return True
    return False


def _should_skip_block(block: dict[str, Any]) -> bool:
    """判断 block 是否应跳过（图表/页眉页脚等非正文内容）。"""
    block_type = str(block.get("type", "")).lower()
    return block_type in _SKIP_BLOCK_TYPES


# ---------------------------------------------------------------------------
# 列表/索引 block 的文本提取
# ---------------------------------------------------------------------------

def _extract_list_text(block: dict[str, Any]) -> str:
    """从 list / index 类型 block 中提取文本。"""
    content_obj = block.get("content")
    if not isinstance(content_obj, dict):
        return ""
    list_items = content_obj.get("list_items")
    if not isinstance(list_items, list):
        return ""
    parts: list[str] = []
    for item in list_items:
        if isinstance(item, dict):
            item_content = item.get("item_content")
            if isinstance(item_content, list):
                for span in item_content:
                    if isinstance(span, dict):
                        c = span.get("content", "")
                        if isinstance(c, str) and c.strip():
                            parts.append(c.strip())
    return " ".join(parts)


def _extract_any_text(block: dict[str, Any]) -> str:
    """统一文本提取入口: 先尝试标准文本键, 再尝试列表。"""
    text = _extract_block_text(block)
    if text:
        return text
    block_type = str(block.get("type", "")).lower()
    if block_type in ("list", "index"):
        return _extract_list_text(block)
    # compatibility with simplified structure
    simple_text = block.get("text")
    if isinstance(simple_text, str) and simple_text.strip():
        return simple_text.strip()
    return ""


# ---------------------------------------------------------------------------
# 层级化树构建：FlatBlock/level 解析
# ---------------------------------------------------------------------------


@dataclass
class FlatBlock:
    page_idx: int
    block_idx: int
    block_type: str  # "title" | "paragraph" | "list" | ...
    text: str
    raw_level: Optional[int]  # only title block
    is_title: bool
    resolved_level: int = 1


def _extract_title_text(block: dict[str, Any]) -> str:
    """提取标题文本（优先 content.title_content，其次兼容简化测试结构）。"""
    content_obj = block.get("content")
    if isinstance(content_obj, dict):
        title_spans = content_obj.get("title_content")
        if isinstance(title_spans, list):
            parts: list[str] = []
            for span in title_spans:
                if isinstance(span, dict):
                    c = span.get("content", "")
                    if isinstance(c, str) and c.strip():
                        parts.append(c.strip())
            if parts:
                return " ".join(parts)
    # compatibility with simplified structure
    text = block.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return ""


def _get_title_raw_level(block: dict[str, Any]) -> Optional[int]:
    """提取原始标题 level（支持真实 v2 结构与简化测试结构）。"""
    content_obj = block.get("content")
    if isinstance(content_obj, dict):
        lvl = content_obj.get("level")
        if isinstance(lvl, int) and lvl >= 1:
            return lvl
    lvl = block.get("text_level", block.get("level"))
    if isinstance(lvl, int) and lvl >= 1:
        return lvl
    return None


def infer_level_by_numbering(title_text: str) -> Optional[int]:
    """从标题文本的编号模式推断层级。

    返回 int (1-4) 或 None (无法推断)。
    规则按优先级排序，先匹配先返回。

    设计原则:
    - 宁可不推断 (None) 也不误推断
    - 只匹配行首编号，避免 "A novel approach" 误命中
    - 层级映射参考学术论文/法律法规/技术文档的通用习惯

    支持的编号与中文正文之间无空格的紧凑格式（中文国标/行标常见）:
    - "1范围", "3术语和定义", "6试验方法"
    - "3.1", "5.1要求内容", "6.1.8.1测量部位"
    """
    text = title_text.strip()
    if not text:
        return None

    # ── P1: 多级数字编号 ──────────────────────────
    # P1a: 多级编号（至少含一个点）+ 空格/中文/拉丁字母/行尾
    #   "2.1 xxx", "2.1.1 xxx", "6.1.8.1测量部位", "3.1", "5.3.1内在质量要求见表1。"
    #   "6.1.3pH值"（编号后直接跟拉丁字母也需匹配）
    m = re.match(r"^(\d+(?:\.\d+)+)\s*\.?\s*(?=[\u4e00-\u9fffa-zA-Z\s(（]|$)", text)
    if m:
        parts = [p for p in m.group(1).split(".") if p]
        return min(len(parts), 4)

    # P1b: 单级编号 + 空格或中文（但排除年份如 "2019年" / "2020年"）
    #   "1范围", "6试验方法", "1 Introduction", "9产品使用说明、包装、运输、贮存"
    #   不匹配: "2019年", "2020-07-01"
    m = re.match(r"^(\d{1,2})\s*\.?\s*(?=[\u4e00-\u9fff(（])", text)
    if m:
        num = int(m.group(1))
        # 排除明显非编号的大数字（年份等）
        if num <= 99:
            return 1

    # P1c: 原有带空格的通用模式（英文文档 "1. Introduction" 等）
    m = re.match(r"^(\d+(?:\.\d+)*)\s*\.?\s(?!\d)", text)
    if m:
        parts = [p for p in m.group(1).split(".") if p]
        return min(len(parts), 4)

    # ── P2: 数字+顿号（中文公文/报告常见）──────────────
    # "1、概述", "2、方法"
    if re.match(r"^\d+、", text):
        return 1

    # ── P3: 数字+右括号（技术文档/列表子项）──────────────
    # "1) Introduction", "2) Methods"
    if re.match(r"^\d+\)\s", text):
        return 2

    # ── P4: 罗马数字（IEEE/法律文档 L1）──────────────
    # 覆盖 I-XXX 的完整罗马数字
    if re.match(
        r"^(I{1,3}|IV|VI{0,3}|IX|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX|V|X)\.\s",
        text,
    ):
        return 1

    # ── P5: 大写字母+点（子章节/附件编号）──────────────
    # "A. First", "B. Second" — 但排除 "A novel..."（需要点+空格）
    if re.match(r"^[A-Z]\.\s", text):
        return 2

    # ── P6: 小写字母+点或右括号（技术文档列表子项）──────
    # "a. Sub section", "a) Sub section"
    if re.match(r"^[a-z][.)]\s", text):
        return 3

    # ── P7: 中文法定编号（法律法规/政策文件）──────────
    if re.match(r"^第.+[章部篇编]", text):
        return 1
    if re.match(r"^第.+[节款]", text):
        return 2
    if re.match(r"^第.+[条项]", text):
        return 3

    # ── P8: 中文大写数字+顿号（政府公文 L1）──────────
    # "一、总述", "二、工作目标"
    if re.match(r"^[一二三四五六七八九十百]+、", text):
        return 1

    # ── P9: 全角/半角括号+数字/中文数字（子级条目）──────
    # "（一）基本原则", "（1）具体方案", "(一) xxx"
    if re.match(r"^[（(][一二三四五六七八九十\d]+[）)]", text):
        return 2

    # ── P10: 半角括号+字母（列表子项）──────────────
    # "(a) First", "(b) Second" — 但排除 (i) (ii) 等小写罗马
    if re.match(r"^\([a-hj-z]\)", text):  # 排除 i 避免罗马数字歧义
        return 3

    # ── P11: 英文结构关键词+编号 ──────────────────
    # "Chapter 1", "Part I", "Part 2", "Section 2.1"
    m = re.match(r"^(?:Chapter|CHAPTER)\s+(\d+)", text, re.IGNORECASE)
    if m:
        return 1
    m = re.match(r"^(?:Part|PART)\s+(\d+|[IVX]+)", text, re.IGNORECASE)
    if m:
        return 1
    m = re.match(r"^(?:Section|SECTION)\s+(\d+(?:\.\d+)*)", text, re.IGNORECASE)
    if m:
        parts = [p for p in m.group(1).split(".") if p]
        return min(len(parts), 4)
    m = re.match(r"^(?:Appendix|APPENDIX|Annex|ANNEX)\s+[A-Z0-9]", text, re.IGNORECASE)
    if m:
        return 1

    # ── P12: §+数字编号（德式法律/ISO 标准）──────────
    # "§1 xxx", "§1.1 xxx"
    m = re.match(r"^§\s*(\d+(?:\.\d+)*)", text)
    if m:
        parts = [p for p in m.group(1).split(".") if p]
        return min(len(parts), 4)

    return None


def resolve_levels(flat_blocks: list[FlatBlock]) -> None:
    """原地更新每个 title FlatBlock 的 resolved_level 字段。

    策略:
    1. 编号 heuristic 能推断 → 直接用 inferred level（最可靠）
    2. heuristic 推断不出 →
       a) 如果文档中存在任何成功推断的编号标题 → 无编号标题默认 L1
          （学术/技术文档中 Abstract/References 等无编号标题是顶级章节）
       b) 如果文档完全无编号标题（纯自然标题）→ 用 raw_level 兜底
          （虽然不精确，但至少保留了相对层级信息，避免全部扁平化）
    """
    title_blocks = [fb for fb in flat_blocks if fb.is_title]

    # 第一遍：跑 heuristic
    inferred_map: dict[int, Optional[int]] = {}  # index in title_blocks → inferred
    has_any_inferred = False
    for i, fb in enumerate(title_blocks):
        inferred = infer_level_by_numbering(fb.text)
        inferred_map[i] = inferred
        if inferred is not None:
            has_any_inferred = True

    # 第二遍：赋值
    for i, fb in enumerate(title_blocks):
        inferred = inferred_map[i]
        if inferred is not None:
            fb.resolved_level = inferred
        elif has_any_inferred:
            # 文档有编号标题，无编号标题视为顶级
            fb.resolved_level = 1
        else:
            # 文档完全无编号 → raw_level 兜底（至少有相对层级）
            fb.resolved_level = fb.raw_level if (fb.raw_level and fb.raw_level >= 1) else 1


def _block_type_to_ref(block: dict[str, Any], is_title: bool) -> str:
    if is_title:
        return "title"
    t = str(block.get("type", "")).lower().strip()
    if t in {"paragraph", "text", "para"}:
        return "paragraph"
    if t in {"list", "index"}:
        return t
    return t or "paragraph"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_section_build_stage(content_list_v2: Any) -> SectionNode:
    """
    SectionBuildStage: 将 content_list_v2 转为层级化 SectionNode tree（纯规则，无 token/LLM）。

    输入:
        content_list_v2 — MinerU 产出的结构化文档，list[list[dict]]
                         外层索引=page_idx，内层每个 dict 是一个 block

    输出:
        SectionNode — root 节点（level=0），其 children 是完整的 N 层 section tree
    """

    # --- 1) 归一化 content_list_v2 到 pages: list[list[block_dict]] ---
    pages: list[tuple[int, list[dict[str, Any]]]] = []
    if isinstance(content_list_v2, list):
        if not content_list_v2:
            pages = []
        else:
            # 如果传入的是 list[dict]（简化结构），视为单页 blocks
            if isinstance(content_list_v2[0], dict):
                grouped: dict[int, list[dict[str, Any]]] = {}
                ordered_fallback: list[dict[str, Any]] = []
                has_explicit_page_idx = False
                for b in content_list_v2:
                    if not isinstance(b, dict):
                        continue
                    raw_page_idx = b.get("page_idx")
                    if isinstance(raw_page_idx, int) and raw_page_idx >= 0:
                        has_explicit_page_idx = True
                        grouped.setdefault(raw_page_idx, []).append(b)
                    else:
                        ordered_fallback.append(b)
                if has_explicit_page_idx:
                    for pidx in sorted(grouped):
                        pages.append((pidx, grouped[pidx]))
                    if ordered_fallback:
                        fallback_start = (max(grouped) + 1) if grouped else 0
                        pages.append((fallback_start, ordered_fallback))
                else:
                    pages = [(0, content_list_v2)]  # type: ignore[list-item]
            else:
                for page_idx, p in enumerate(content_list_v2):
                    if isinstance(p, list):
                        pages.append((page_idx, [b for b in p if isinstance(b, dict)]))

    # --- 2) 扁平化：收集所有有效文本 block ---
    flat_blocks: list[FlatBlock] = []
    for page_idx, page_blocks in pages:
        for block_idx, block in enumerate(page_blocks):
            if not isinstance(block, dict):
                continue
            if _should_skip_block(block):
                continue

            is_title = _is_title_block(block)
            if is_title:
                title_text = _extract_title_text(block)
                raw_level = _get_title_raw_level(block)
                fb = FlatBlock(
                    page_idx=page_idx,
                    block_idx=block_idx,
                    block_type="title",
                    text=title_text,
                    raw_level=raw_level,
                    is_title=True,
                )
                # title 提取失败：按文档兜底命名
                if not fb.text.strip():
                    fb.text = f"(Untitled page {page_idx})"
                flat_blocks.append(fb)
            else:
                text = _extract_any_text(block)
                if not text.strip():
                    continue
                fb = FlatBlock(
                    page_idx=page_idx,
                    block_idx=block_idx,
                    block_type=_block_type_to_ref(block, is_title=False),
                    text=text,
                    raw_level=None,
                    is_title=False,
                )
                flat_blocks.append(fb)

    # --- 3) 解析真实 level（heuristic 优先，raw_level 兜底）---
    resolve_levels(flat_blocks)

    # --- 4) 栈构建 tree ---
    root = SectionNode(
        section_id="doc_root",
        title="",
        level=0,
        own_text="",
        page_range=[-1, -1],
        block_refs=[],
        children=[],
    )

    stack: list[SectionNode] = [root]
    text_buffer: list[str] = []
    ref_buffer: list[dict[str, Any]] = []
    page_buffer: list[int] = []
    id_counter = 0
    prologue_emitted = False  # 标记是否已处理过 title 前的文本

    def next_id() -> str:
        nonlocal id_counter
        id_counter += 1
        return f"sec_{id_counter:03d}"

    def flush_text_to_top() -> None:
        nonlocal text_buffer, ref_buffer, page_buffer
        if not text_buffer:
            return

        top = stack[-1]
        new_text = "\n".join(text_buffer).strip()
        if new_text:
            if top.own_text:
                top.own_text += "\n" + new_text
            else:
                top.own_text = new_text
            top.block_refs.extend(ref_buffer)

            if page_buffer:
                if top.page_range == [-1, -1]:
                    top.page_range = [min(page_buffer), max(page_buffer)]
                else:
                    top.page_range[0] = min(top.page_range[0], min(page_buffer))
                    top.page_range[1] = max(top.page_range[1], max(page_buffer))

        text_buffer = []
        ref_buffer = []
        page_buffer = []

    def _emit_prologue_if_needed() -> None:
        """遇到第一个 title 时，如果 root 上已累积了文本，立即创建 prologue 节点。

        这样 prologue 的 id 按文档顺序排在所有 title 节点之前。
        """
        nonlocal prologue_emitted
        if prologue_emitted:
            return
        prologue_emitted = True

        # 先把 buffer 刷到 root（此时 stack 顶是 root）
        flush_text_to_top()

        if root.own_text.strip():
            prologue = SectionNode(
                section_id=next_id(),
                title="(Untitled Prologue)",
                level=1,
                own_text=root.own_text,
                page_range=list(root.page_range),
                block_refs=list(root.block_refs),
                children=[],
            )
            root.children.append(prologue)
            root.own_text = ""
            root.page_range = [-1, -1]
            root.block_refs = []

    for fb in flat_blocks:
        if fb.is_title:
            level = fb.resolved_level

            # 第一个 title 之前的文本 → 立即创建 prologue（id 按文档顺序）
            _emit_prologue_if_needed()

            flush_text_to_top()

            # 弹出所有 level >= 当前标题的栈元素
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()

            new_node = SectionNode(
                section_id=next_id(),
                title=fb.text.strip(),
                level=level,
                own_text="",
                page_range=[fb.page_idx, fb.page_idx],
                block_refs=[
                    {
                        "page_idx": fb.page_idx,
                        "block_idx": fb.block_idx,
                        "type": "title",
                    }
                ],
                children=[],
            )

            stack[-1].children.append(new_node)
            stack.append(new_node)
        else:
            text = fb.text.strip()
            if text:
                text_buffer.append(text)
                ref_buffer.append(
                    {
                        "page_idx": fb.page_idx,
                        "block_idx": fb.block_idx,
                        "type": fb.block_type,
                    }
                )
                page_buffer.append(fb.page_idx)

    flush_text_to_top()

    # --- 5) orphan text 后处理 ---
    # prologue 已在主循环中处理（_emit_prologue_if_needed）。
    # 这里只处理两种边缘情况：
    #   a) 文档无任何 title（纯文本）→ root.own_text 非空但无 children
    #   b) 文档完全为空 → root.own_text 为空且无 children
    if root.own_text.strip() and root.children:
        # 不应发生（prologue 已在循环中处理），但防御性处理
        prologue = SectionNode(
            section_id=next_id(),
            title="(Untitled Prologue)",
            level=1,
            own_text=root.own_text,
            page_range=list(root.page_range),
            block_refs=list(root.block_refs),
            children=[],
        )
        root.children.insert(0, prologue)
        root.own_text = ""
        root.page_range = [-1, -1]
        root.block_refs = []
    elif not root.children:
        if root.own_text.strip():
            sole = SectionNode(
                section_id=next_id(),
                title="(Untitled Document)",
                level=1,
                own_text=root.own_text,
                page_range=list(root.page_range),
                block_refs=list(root.block_refs),
                children=[],
            )
            root.children.append(sole)
            root.own_text = ""
            root.page_range = [-1, -1]
            root.block_refs = []
        else:
            root.children.append(
                SectionNode(
                    section_id=next_id(),
                    title="(Empty Document)",
                    level=1,
                    own_text="",
                    page_range=[-1, -1],
                    block_refs=[],
                    children=[],
                )
            )

    def _merge_child_page_ranges(node: SectionNode) -> None:
        child_pages: list[int] = []
        for c in node.children:
            _merge_child_page_ranges(c)
            if c.page_range != [-1, -1]:
                child_pages.extend(c.page_range)
        if child_pages:
            if node.page_range == [-1, -1]:
                node.page_range = [min(child_pages), max(child_pages)]
            else:
                node.page_range = [
                    min(node.page_range[0], min(child_pages)),
                    max(node.page_range[1], max(child_pages)),
                ]

    _merge_child_page_ranges(root)

    # 合约收尾
    root.own_text = ""
    root.title = ""
    root.level = 0
    root.page_range = [-1, -1]

    return root
