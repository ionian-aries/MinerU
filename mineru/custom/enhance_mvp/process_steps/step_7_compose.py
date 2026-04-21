"""Stage 7: ComposeStage — 将增强结果插入 markdown 输出。

注入位置规则：
- overview：文档头部，split_marker（来自 chunking.split_marker 配置）紧跟在 </enhance> 下方。
- section <enhance>：注入在对应标题行之前，split_marker 在 <enhance> 之上。
- 图片描述：紧跟在含图片文件名的行之后（引用块格式）。
- 兜底未匹配 section：追加到文档末尾，split_marker 在 <enhance> 之上。

split_marker 为空字符串时不注入分段标记，行为与旧版一致。

使用归一化后的模糊匹配替代精确匹配，解决 markdown 转义字符导致标题匹配失败的问题。

输出格式（@notes/14 §7.3）:
- overview:  <enhance type="overview"> + 全文摘要/关键词/洞察提纲（含 [section_id]）
- section:   <enhance id="sec_xxx" pages="0-2"> + 人可读 key:value body
- refs 不在 md 中出现（仅在 enhance.json 中）
- 失败的 section 不输出 enhance 标签
"""
from __future__ import annotations

import collections
import re
import unicodedata
from typing import Any

from loguru import logger


def _normalize_for_match(text: str) -> str:
    """将文本归一化用于模糊匹配。"""
    s = text.strip().lstrip("#").strip()
    s = re.sub(r"\\([*$`~\[\](){}!|>#+\\-_.])", r"\1", s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.lower()
    return s


def _format_pages(page_range: Any) -> str:
    """[0, 2] → "0-2", [-1,-1] → ""."""
    if not isinstance(page_range, (list, tuple)) or len(page_range) < 2:
        return ""
    start, end = page_range[0], page_range[1]
    if start < 0 or end < 0:
        return ""
    return f"{start}-{end}"


def _render_overview(overview: dict[str, Any]) -> list[str]:
    """渲染 overview 为人可读 key:value 格式（仅读取 global_* 与 doc_outline_insights）。"""
    lines: list[str] = ['<enhance type="overview">']

    summary = str(overview.get("global_summary", "")).strip()
    if summary:
        lines.append(f"全文摘要: {summary}")

    keywords = overview.get("global_keywords")
    if isinstance(keywords, list) and keywords:
        lines.append(f"关键词: {', '.join(str(k) for k in keywords)}")

    insights = overview.get("doc_outline_insights")
    if isinstance(insights, list) and insights:
        lines.append("洞察提纲:")
        for item in insights:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("section_id", "")).strip()
            text = str(item.get("text", "")).strip()
            text = re.sub(r"^\d+\.\s*", "", text)
            if not text or not sid:
                continue
            lines.append(f"- [{sid}] {text}")

    answerable_qs = overview.get("answerable_questions")
    if isinstance(answerable_qs, list) and answerable_qs:
        lines.append("可回答问题:")
        for q in answerable_qs:
            if str(q).strip():
                lines.append(f"- {str(q).strip()}")

    lines.append("</enhance>")
    return lines


def _render_section(n: dict[str, Any]) -> list[str]:
    """渲染单个 section 节点为人可读 key:value 格式。

    只渲染 status=success 的节点。
    merged 单元输出 source 属性标识被合并的原始 section_ids。
    """
    if n.get("status") != "success":
        return []

    sec_id = str(n.get("id", "sec_unknown"))
    pages = _format_pages(n.get("page_range"))

    attrs = f'id="{sec_id}"'
    if pages:
        attrs += f' pages="{pages}"'

    source_ids = n.get("source_section_ids")
    if n.get("merge_type") == "merged" and isinstance(source_ids, list) and len(source_ids) > 1:
        attrs += f' source="{",".join(source_ids)}"'

    lines: list[str] = [f"<enhance {attrs}>"]

    summary = str(n.get("summary", "")).strip()
    if summary:
        lines.append(f"摘要: {summary}")

    keywords = n.get("keywords", [])
    if isinstance(keywords, list) and keywords:
        lines.append(f"关键词: {', '.join(str(k) for k in keywords)}")

    main_idea = str(n.get("main_idea", "")).strip()
    if main_idea:
        lines.append(f"核心观点: {main_idea}")

    questions = n.get("questions", [])
    if isinstance(questions, list) and questions:
        lines.append("可回答问题:")
        for q in questions:
            if str(q).strip():
                lines.append(f"- {str(q).strip()}")

    lines.append("</enhance>")
    return lines


def _walk_nodes(node_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """展平 section 结果为节点队列（preorder）。"""
    out: list[dict[str, Any]] = []
    for n in node_list:
        out.append(n)
        subs = n.get("subsections")
        if isinstance(subs, list) and subs:
            out.extend(_walk_nodes(subs))
    return out


def run_compose_stage(
    raw_md: str,
    enhancement_payload: dict[str, Any],
    *,
    split_marker: str = "",
) -> str:
    """将增强结果插入 markdown 输出。

    注入位置规则：
    - overview：文档头部，split_marker 紧跟在 </enhance> 之后。
    - section <enhance>：注入在对应标题行之前，split_marker 在 <enhance> 之上。
    - 图片描述：紧跟在含图片文件名的行之后（引用块格式）。
    - 兜底未匹配 section：追加到文档末尾，split_marker 在 <enhance> 之上。

    split_marker 为空字符串时不注入分段标记。
    """
    overview = enhancement_payload.get("overview", {}) or {}
    sections = enhancement_payload.get("sections", []) or []

    all_nodes = _walk_nodes(sections) if isinstance(sections, list) else []
    logger.debug(f"[ComposeStage] start: sections={len(all_nodes)}")

    # 按锚点标题建索引：normalized_title → deque[section_node]
    title_to_queue: dict[str, collections.deque[dict[str, Any]]] = {}
    for n in all_nodes:
        title = str(n.get("anchor_title") or n.get("title", "")).strip()
        if not title:
            continue
        norm = _normalize_for_match(title)
        if not norm:
            continue
        if norm not in title_to_queue:
            title_to_queue[norm] = collections.deque()
        title_to_queue[norm].append(n)

    # 图片描述索引：img_path（裸文件名）→ description（仅 status=success 且有描述的）
    image_enhancements: dict[str, Any] = enhancement_payload.get("image_enhancements") or {}
    img_desc_map: dict[str, str] = {
        img_path: str(rec["description"])
        for img_path, rec in image_enhancements.items()
        if isinstance(rec, dict) and rec.get("status") == "success" and rec.get("description")
    }
    img_injected: set[str] = set()

    consumed: set[str] = set()

    # ---- 组装 markdown 输出 ----
    lines: list[str] = []

    # overview 注入：头部，split_marker 在 </enhance> 下方
    lines.extend(_render_overview(overview))
    if split_marker:
        lines.append(split_marker)
    lines.append("")

    source_lines = (raw_md or "").rstrip().splitlines()
    for line in source_lines:
        stripped = line.strip()

        # 标题行：先在标题前注入 split_marker + <enhance>，再输出标题行本身
        if stripped.startswith("#"):
            norm_line = _normalize_for_match(stripped)
            if norm_line:
                q = title_to_queue.get(norm_line)
                if q:
                    while q and str(q[0].get("id", "")) in consumed:
                        q.popleft()
                    if q:
                        node = q.popleft()
                        sid = str(node.get("id", ""))
                        if sid not in consumed:
                            rendered = _render_section(node)
                            if rendered:
                                if split_marker:
                                    lines.append(split_marker)
                                lines.extend(rendered)
                            consumed.add(sid)

        lines.append(line)

        # 图片描述注入：在含图片文件名的行后插入描述（引用块格式）
        for img_path, desc in img_desc_map.items():
            if img_path not in img_injected and img_path in line:
                lines.append(f"> **图片描述**：{desc}")
                img_injected.add(img_path)
                break

    matched_count = len(consumed)

    # ---- 未匹配到标题行的 section 追加到文档末尾 ----
    for n in all_nodes:
        sid = str(n.get("id", ""))
        if not sid or sid in consumed:
            continue
        rendered = _render_section(n)
        if rendered:
            if split_marker:
                lines.append(split_marker)
            lines.extend(rendered)
        consumed.add(sid)

    logger.info(
        f"[ComposeStage] done: sections={len(all_nodes)} "
        f"matched={matched_count} appended={len(consumed) - matched_count}"
    )
    return "\n".join(lines).rstrip() + "\n"
