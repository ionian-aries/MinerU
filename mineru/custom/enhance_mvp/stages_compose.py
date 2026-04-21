"""Stage 7: ComposeStage — 将增强结果插入 markdown 输出。

每个 section 的 <enhance> 标签插入到其对应的 markdown 标题行之后（扁平放置）。
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

    # merged 单元：添加 source 属性
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


def run_compose_stage(raw_md: str, enhancement_payload: dict[str, Any]) -> str:
    """将增强结果插入 markdown 输出。

    每个 section 的 <enhance> 标签紧跟在其对应标题行之后。
    不做嵌套：父节点和子节点各自独立放置在各自的标题位置。
    """
    overview = enhancement_payload.get("overview", {}) or {}
    sections = enhancement_payload.get("sections", []) or []

    all_nodes = _walk_nodes(sections) if isinstance(sections, list) else []

    # 按锚点标题建索引：normalized_title → deque[section_node]
    # 优先使用 anchor_title（原始标题），否则回退到 title（LLM 合并标题）。
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

    consumed: set[str] = set()

    # ---- 组装 markdown 输出 ----
    lines: list[str] = []
    lines.extend(_render_overview(overview))
    lines.append("")

    source_lines = (raw_md or "").rstrip().splitlines()
    for line in source_lines:
        lines.append(line)
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue

        norm_line = _normalize_for_match(stripped)
        if not norm_line:
            continue

        q = title_to_queue.get(norm_line)
        if not q:
            continue

        # 跳过已消费的节点
        while q and str(q[0].get("id", "")) in consumed:
            q.popleft()
        if not q:
            continue

        node = q.popleft()
        sid = str(node.get("id", ""))
        if sid in consumed:
            continue

        rendered = _render_section(node)
        if rendered:
            lines.extend(rendered)
        consumed.add(sid)

    # ---- 未匹配到标题行的 section 追加到文档末尾 ----
    for n in all_nodes:
        sid = str(n.get("id", ""))
        if not sid or sid in consumed:
            continue
        rendered = _render_section(n)
        if rendered:
            lines.append("")
            lines.extend(rendered)
        consumed.add(sid)

    return "\n".join(lines).rstrip() + "\n"
