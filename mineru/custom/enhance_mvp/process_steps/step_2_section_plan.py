"""Stage 2: Hybrid Chunk Planner — 基于结构感知与语义理解的智能分块。

输入：Block Indexer 产出的 list[TitleItem | BodyItem]（flat 物理顺序）
输出：list[EnhanceUnit]（flat，无树结构）+ SectionPlanResult

核心设计：
  Phase A: 解析 items → 层级大纲（短正文内联，长正文 [bN] 块列表）
  Phase B: Token 预算估算 → 路由决策（单次调用 vs 两级规划）
  Phase C: LLM 规划调用
    - 预算充足：单次 C2 详细规划
    - 预算不足：C1 骨架规划 → 并发 C2 章节规划（失败则退化到滑动窗口）
  Phase D: 三层校验 + 自动修复（存在性 → 连续性 → 覆盖性）
  Phase E: EnhanceUnit 组装
    - split_at → 按 block_items 切片生成子块（精确 ref 归属）
    - 超长兜底 → 段落对齐分割

关键约束：
  - section_id 在校验通过后才分配（s001, s002, …）
  - 子块 section_id = "{parent_id}p{N:02d}"（如 s003p01, s003p02）
  - estimated_tokens / char_count 在组装阶段 inline 计算（Stage 3 passthrough）
  - content_description：自包含一句话，供 RAG 前缀使用
  - split_at：仅对 seq_ids 单元素的节有效，值为 block_items 切分点
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from loguru import logger

from mineru.custom.enhance_mvp.prompts import render_prompt
from mineru.custom.enhance_mvp.schema import EnhanceUnit
from mineru.custom.enhance_mvp.utils import TokenCounter

from .step_1_build_block_items import BlockUnit, BodyItem, TitleItem

# 默认字符数约束
_DEFAULT_CHUNK_MAX_CHARS = 3000
_DEFAULT_CHUNK_MIN_CHARS = 200

# 正文内联展示阈值：body 字符数 ≤ 此值时内联显示文本，否则显示 [bN] 块列表
_INLINE_MAX_CHARS = 400

# C1 骨架规划的 system prompt（内联，不需要模板）
_C1_SYSTEM_PROMPT = (
    '你是文档结构分析专家。你的任务是将文档标题列表划分为若干"章节组"（chapter），'
    "每组包含语义相关的连续标题集合。章节组是后续详细规划的基础，不需要过于细粒度。"
    "优先按文档的一级章节边界划分，每组包含 5-30 个标题节。"
    "必须返回 JSON 对象。"
)


# ---------------------------------------------------------------------------
# 结果元数据
# ---------------------------------------------------------------------------

@dataclass
class SectionPlanResult:
    """SectionPlanStage 结果元数据，写入 enhance.json 的 section_plan 段。"""

    strategy: str      # "llm_assisted" | "skipped" | "fallback"
    doc_type: str
    fragmentation_rate: float
    original_section_count: int
    enhance_unit_count: int
    llm_calls: int
    fallback: bool
    fallback_reason: str = ""


# ---------------------------------------------------------------------------
# Phase A: 大纲构建
# ---------------------------------------------------------------------------

def _fmt_block(bu: BlockUnit, idx: int) -> str:
    """将单个 BlockUnit 格式化为 [bN] 条目（长正文块列表用）。"""
    if bu.image_count:
        return f"  [b{idx}] 图片/图表"
    if bu.table_count:
        return f"  [b{idx}] 表格"
    preview = bu.rendered_text[:50].replace("\n", " ").strip()
    suffix = "…" if bu.char_count > 50 else ""
    return f"  [b{idx}] 文本（{bu.char_count}字）" + (f"：{preview}{suffix}" if preview else "")


def _fmt_body_inline(body: BodyItem, inline_max_chars: int) -> str:
    """格式化 BodyItem：短正文内联，长正文显示 [bN] 块列表。"""
    if body.char_count == 0:
        return ""
    if body.char_count <= inline_max_chars:
        text = body.body_text[:inline_max_chars].replace("\n", " ").strip()
        return f"  > {text}"
    n = len(body.block_items)
    lines = [f"  正文（{body.char_count}字，{n}个块）："]
    for i, bu in enumerate(body.block_items):
        lines.append(_fmt_block(bu, i))
    return "\n".join(lines)


def _build_outline(
    title_items: list[TitleItem],
    body_map: dict[int, BodyItem],
    preamble: BodyItem | None,
    inline_max_chars: int = _INLINE_MAX_CHARS,
    title_offset: int = 0,
) -> str:
    """构建层级大纲（C2 详细规划输入）。

    短正文（≤ inline_max_chars）内联显示；长正文显示 [bN] 块列表。
    """
    lines: list[str] = []

    if preamble and preamble.char_count > 0:
        lines.append(f"[前言]（{preamble.char_count}字）")
        body_fmt = _fmt_body_inline(preamble, inline_max_chars)
        if body_fmt:
            lines.append(body_fmt)

    for local_idx, t in enumerate(title_items):
        global_idx = title_offset + local_idx
        body = body_map.get(local_idx)
        body_chars = body.char_count if body else 0

        if body_chars == 0:
            lines.append(f"[{global_idx:02d}] {t.title_text}（无正文）")
        else:
            lines.append(f"[{global_idx:02d}] {t.title_text}（{body_chars}字）")
            if body:
                body_fmt = _fmt_body_inline(body, inline_max_chars)
                if body_fmt:
                    lines.append(body_fmt)

    return "\n".join(lines)


def _build_skeleton(
    title_items: list[TitleItem],
    body_map: dict[int, BodyItem],
    preamble: BodyItem | None,
) -> str:
    """构建骨架大纲（C1 全局规划输入）：仅标题 + 字符数，无正文内容。"""
    lines: list[str] = []
    if preamble and preamble.char_count > 0:
        lines.append(f"[前言]（{preamble.char_count}字正文）")
    for idx, t in enumerate(title_items):
        body = body_map.get(idx)
        body_chars = body.char_count if body else 0
        lines.append(f"[{idx:02d}] {t.title_text}（{body_chars}字正文）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase D: 三层校验 + 自动修复
# ---------------------------------------------------------------------------

def _validate_and_fix(
    raw_sections: list[dict[str, Any]],
    max_seq_id: int,
) -> tuple[list[dict[str, Any]], bool]:
    """三层校验并自动修复 LLM 输出的 sections 规划。

    Layer 1 存在性：所有 seq_id ∈ [0, max_seq_id]
    Layer 2 连续性：每个 section 内 seq_ids = list(range(min, max+1))
    Layer 3 覆盖性：所有 sections 的 seq_ids 并集 = [0..max_seq_id] 全集

    返回: (fixed_sections, has_fatal_error)
    fixed_sections 中每个元素保证：
        {seq_ids, title, content_description, enhance_guidance, rationale, split_at(可选)}
    """
    if not isinstance(raw_sections, list) or not raw_sections:
        return [], True

    valid_range = set(range(max_seq_id + 1))
    used_ids: set[int] = set()
    fixed: list[dict[str, Any]] = []

    for item in raw_sections:
        if not isinstance(item, dict):
            continue
        seq_ids_raw = item.get("seq_ids")
        if not isinstance(seq_ids_raw, list) or not seq_ids_raw:
            continue

        # Layer 1: 过滤非法 seq_id
        valid_ids = [
            int(sid) for sid in seq_ids_raw
            if isinstance(sid, (int, float)) and int(sid) in valid_range and int(sid) not in used_ids
        ]
        if not valid_ids:
            continue

        # Layer 2: 连续性修复（不连续 → 拆为多个连续段）
        valid_ids_sorted = sorted(valid_ids)
        groups: list[list[int]] = [[valid_ids_sorted[0]]]
        for sid in valid_ids_sorted[1:]:
            if sid == groups[-1][-1] + 1:
                groups[-1].append(sid)
            else:
                groups.append([sid])

        title = str(item.get("title", "")).strip() or "章节单元"
        rationale = str(item.get("rationale", "")).strip() or "语义分组"
        enhance_guidance = str(item.get("enhance_guidance", "") or "").strip()
        content_description = str(item.get("content_description", "") or "").strip()

        # split_at 仅保留有效整数（连续性拆分后 split_at 仅对单元素组有意义，其他组忽略）
        raw_split_at = item.get("split_at")
        split_at: list[int] = []
        if isinstance(raw_split_at, list):
            split_at = [int(x) for x in raw_split_at if isinstance(x, (int, float)) and int(x) > 0]

        for grp in groups:
            if not grp:
                continue
            entry: dict[str, Any] = {
                "seq_ids": grp,
                "title": title,
                "content_description": content_description,
                "enhance_guidance": enhance_guidance,
                "rationale": rationale,
            }
            # split_at 仅在单元素组时保留
            if split_at and len(grp) == 1:
                entry["split_at"] = split_at
            fixed.append(entry)
            used_ids.update(grp)

    # Layer 3: 覆盖性修复（自动补全未覆盖的 seq_id）
    missing = valid_range - used_ids
    if missing:
        logger.debug(f"[SectionPlanStage] coverage repair: missing seq_ids={sorted(missing)}")
        for sid in sorted(missing):
            fixed.append({
                "seq_ids": [sid],
                "title": f"章节{sid:02d}",
                "content_description": "",
                "enhance_guidance": "",
                "rationale": "自动补全",
            })
            used_ids.add(sid)

    return fixed, False


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _compute_page_range(block_refs: list[list]) -> list[int]:
    pages = [r[0] for r in block_refs if isinstance(r, list) and len(r) >= 1]
    if not pages:
        return [-1, -1]
    return [min(pages), max(pages)]


def _parse_items(
    items: list[TitleItem | BodyItem],
) -> tuple[list[TitleItem], dict[int, BodyItem], BodyItem | None]:
    title_items: list[TitleItem] = []
    body_map: dict[int, BodyItem] = {}
    preamble: BodyItem | None = None
    title_idx = -1

    for item in items:
        if isinstance(item, TitleItem):
            title_idx += 1
            title_items.append(item)
        elif isinstance(item, BodyItem):
            if title_idx < 0:
                preamble = item
            else:
                body_map[title_idx] = item

    return title_items, body_map, preamble


# ---------------------------------------------------------------------------
# Phase E: EnhanceUnit 组装
# ---------------------------------------------------------------------------

def _build_sub_chunks_from_split(
    section: dict[str, Any],
    seq_ids: list[int],
    title_items: list[TitleItem],
    body_map: dict[int, BodyItem],
    preamble: BodyItem | None,
    section_id: str,
    token_counter: TokenCounter | None,
) -> list[EnhanceUnit] | None:
    """处理 split_at：按 block_items 切片生成子块 EnhanceUnit。

    split_at = [3] 表示在第 3 块（[b3]）之前切分：
      子块1 = block_items[0:3]
      子块2 = block_items[3:]

    只对 seq_ids 单元素的节有效（多节合并后不支持）。
    返回 None 表示无需 split_at 处理（交给普通组装）。
    """
    split_at_raw: list[int] = sorted(set(int(x) for x in (section.get("split_at") or [])))
    if not split_at_raw:
        return None
    if len(seq_ids) != 1:
        # 多节合并不支持 split_at
        return None

    seq_id = seq_ids[0]
    body = body_map.get(seq_id)
    # 边界：第一个节且有前言但无 body_map 记录
    if body is None and seq_id == 0 and preamble and preamble.char_count > 0:
        body = preamble

    if body is None or not body.block_items:
        return None

    n_blocks = len(body.block_items)
    valid_splits = sorted(i for i in split_at_raw if 0 < i < n_blocks)
    if not valid_splits:
        return None

    t = title_items[seq_id] if seq_id < len(title_items) else None
    title_ref = [t.block_ref] if t else []
    anchor_title = t.title_text if t else ""
    section_title = str(section.get("title", "") or anchor_title)
    enhance_guidance = str(section.get("enhance_guidance", "") or "")
    content_description = str(section.get("content_description", "") or "")

    boundaries = [0] + valid_splits + [n_blocks]
    n_parts = len(boundaries) - 1
    sub_units: list[EnhanceUnit] = []

    for chunk_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        chunk_blocks = body.block_items[start:end]

        chunk_text = "".join(
            bu.rendered_text + "\n\n" for bu in chunk_blocks if bu.rendered_text
        ).strip()

        chunk_refs: list[list] = []
        if chunk_idx == 0:
            chunk_refs.extend(title_ref)
        for bu in chunk_blocks:
            chunk_refs.extend(bu.block_refs)

        sub_id = f"{section_id}p{chunk_idx + 1:02d}"
        sub_title = f"{section_title}（第{chunk_idx + 1}部分/共{n_parts}部分）"
        char_count = len(chunk_text)

        sub_units.append(EnhanceUnit(
            section_id=sub_id,
            title=sub_title,
            anchor_title=anchor_title,
            seq_ids=seq_ids,
            body_text=chunk_text,
            block_refs=chunk_refs,
            page_range=_compute_page_range(chunk_refs),
            enhance_guidance=enhance_guidance,
            content_description=content_description,
            char_count=char_count,
            estimated_tokens=(token_counter.count(chunk_text) if token_counter else char_count // 2),
            image_count=sum(bu.image_count for bu in chunk_blocks),
            table_count=sum(bu.table_count for bu in chunk_blocks),
            parent_section_id=section_id,
            sub_chunk_index=chunk_idx + 1,
        ))

    logger.debug(
        f"[SectionPlanStage] split_at {section_id}: {n_blocks} blocks "
        f"→ {n_parts} sub-chunks at {valid_splits}"
    )
    return sub_units


def build_enhance_units(
    validated_sections: list[dict[str, Any]],
    title_items: list[TitleItem],
    body_map: dict[int, BodyItem],
    preamble: BodyItem | None,
    token_counter: TokenCounter | None = None,
) -> list[EnhanceUnit]:
    """根据校验通过的 sections 规划，构建 EnhanceUnit 列表。

    section_id 在此处分配（s001, s002, …）。
    优先处理 split_at（按 block_items 切片生成子块）；
    无 split_at 则普通组装（合并多节时插入子标题）。
    """
    units: list[EnhanceUnit] = []

    for i, section in enumerate(validated_sections):
        section_id = f"s{i + 1:03d}"
        seq_ids: list[int] = sorted(section["seq_ids"])

        # 尝试 split_at 子块切分
        sub_chunks = _build_sub_chunks_from_split(
            section, seq_ids, title_items, body_map, preamble, section_id, token_counter,
        )
        if sub_chunks:
            units.extend(sub_chunks)
            continue

        # 普通组装（无 split_at）
        block_refs: list[list] = []
        body_text_parts: list[str] = []
        total_image_count = 0
        total_table_count = 0

        if seq_ids and seq_ids[0] == 0 and preamble is not None:
            block_refs.extend(preamble.block_refs)
            if preamble.body_text.strip():
                body_text_parts.append(preamble.body_text.strip())

        is_merged = len(seq_ids) > 1

        for idx in seq_ids:
            t = title_items[idx] if idx < len(title_items) else None
            if t is None:
                continue
            block_refs.append(t.block_ref)
            body = body_map.get(idx)
            if body:
                block_refs.extend(body.block_refs)
                if body.body_text.strip():
                    if is_merged:
                        # 合并节：插入子标题，帮助 Stage 4 LLM 理解结构
                        body_text_parts.append(f"### {t.title_text}\n{body.body_text.strip()}")
                    else:
                        body_text_parts.append(body.body_text.strip())
                total_image_count += body.image_count
                total_table_count += body.table_count

        body_text = "\n\n".join(body_text_parts)
        char_count = len(body_text)
        estimated_tokens = token_counter.count(body_text) if token_counter else (char_count // 2)

        unit_title = str(section.get("title", "") or "").strip()
        if not unit_title and seq_ids:
            first_t = title_items[seq_ids[0]] if seq_ids[0] < len(title_items) else None
            unit_title = first_t.title_text if first_t else f"章节{seq_ids[0]:02d}"

        anchor_title = ""
        if seq_ids:
            first_t = title_items[seq_ids[0]] if seq_ids[0] < len(title_items) else None
            anchor_title = (first_t.title_text if first_t else "").strip()

        units.append(EnhanceUnit(
            section_id=section_id,
            title=unit_title,
            anchor_title=anchor_title,
            seq_ids=seq_ids,
            body_text=body_text,
            block_refs=block_refs,
            page_range=_compute_page_range(block_refs),
            enhance_guidance=str(section.get("enhance_guidance", "") or ""),
            content_description=str(section.get("content_description", "") or ""),
            char_count=char_count,
            estimated_tokens=estimated_tokens,
            image_count=total_image_count,
            table_count=total_table_count,
        ))

    return units


# ---------------------------------------------------------------------------
# 超大分块兜底拆分（段落对齐）
# ---------------------------------------------------------------------------

def _split_oversized_units(
    units: list[EnhanceUnit],
    chunk_max_chars: int,
    token_counter: "TokenCounter | None" = None,
) -> list[EnhanceUnit]:
    """将 char_count > chunk_max_chars 的 EnhanceUnit 按段落边界拆分。

    这是 split_at 之后的最后兜底：当 LLM 未使用 split_at 但结果仍超长时触发。
    按双换行符（段落边界）贪心分组，而非任意字符截断，保证语义完整。
    子块 section_id = "{original_id}p{N:02d}"，共享 anchor_title、block_refs、page_range。
    """
    result: list[EnhanceUnit] = []

    for unit in units:
        if unit.char_count <= chunk_max_chars:
            result.append(unit)
            continue

        # 按段落边界贪心分割
        paragraphs = unit.body_text.split("\n\n")
        chunks: list[str] = []
        current: list[str] = []
        current_chars = 0

        for para in paragraphs:
            para_chars = len(para)
            if current and current_chars + para_chars + 2 > chunk_max_chars:
                chunks.append("\n\n".join(current))
                current = [para]
                current_chars = para_chars
            else:
                current.append(para)
                current_chars += para_chars + 2

        if current:
            chunks.append("\n\n".join(current))

        if len(chunks) <= 1:
            result.append(unit)
            continue

        n_parts = len(chunks)
        logger.debug(
            f"[SectionPlanStage] oversized fallback split {unit.section_id} "
            f"({unit.char_count} chars) → {n_parts} paragraph-aligned parts"
        )

        parent_id = unit.section_id
        for j, chunk_text in enumerate(chunks):
            sub_id = f"{parent_id}p{j + 1:02d}"
            sub_title = f"{unit.title}（第{j + 1}部分/共{n_parts}部分）"
            char_count = len(chunk_text)
            result.append(EnhanceUnit(
                section_id=sub_id,
                title=sub_title,
                anchor_title=unit.anchor_title,
                seq_ids=unit.seq_ids,
                body_text=chunk_text,
                block_refs=unit.block_refs,
                page_range=unit.page_range,
                enhance_guidance=unit.enhance_guidance,
                content_description=unit.content_description,
                char_count=char_count,
                estimated_tokens=(
                    token_counter.count(chunk_text) if token_counter else char_count // 2
                ),
                image_count=unit.image_count if j == 0 else 0,
                table_count=unit.table_count if j == 0 else 0,
                parent_section_id=parent_id,
                sub_chunk_index=j + 1,
            ))

    return result


# ---------------------------------------------------------------------------
# Fallback：每个 TitleItem 独立成节
# ---------------------------------------------------------------------------

def _fallback_each_title_standalone(
    title_items: list[TitleItem],
    body_map: dict[int, BodyItem],
    preamble: BodyItem | None,
    token_counter: TokenCounter | None = None,
) -> list[EnhanceUnit]:
    if not title_items:
        block_refs: list[list] = []
        body_text = ""
        if preamble:
            block_refs.extend(preamble.block_refs)
            body_text = preamble.body_text.strip()
        char_count = len(body_text)
        return [EnhanceUnit(
            section_id="s001",
            title="全文内容",
            anchor_title="",
            seq_ids=[],
            body_text=body_text,
            block_refs=block_refs,
            page_range=_compute_page_range(block_refs),
            enhance_guidance="",
            content_description="",
            char_count=char_count,
            estimated_tokens=token_counter.count(body_text) if token_counter else char_count // 2,
        )]

    units: list[EnhanceUnit] = []
    for i, t in enumerate(title_items):
        section_id = f"s{i + 1:03d}"
        block_refs = [t.block_ref]
        body = body_map.get(i)
        if body:
            block_refs.extend(body.block_refs)
        if i == 0 and preamble:
            block_refs = list(preamble.block_refs) + block_refs
        body_text = body.body_text.strip() if body else ""
        char_count = len(body_text)
        units.append(EnhanceUnit(
            section_id=section_id,
            title=t.title_text,
            anchor_title=t.title_text,
            seq_ids=[i],
            body_text=body_text,
            block_refs=block_refs,
            page_range=_compute_page_range(block_refs),
            enhance_guidance="",
            content_description="",
            char_count=char_count,
            estimated_tokens=token_counter.count(body_text) if token_counter else char_count // 2,
            image_count=body.image_count if body else 0,
            table_count=body.table_count if body else 0,
        ))
    return units


# ---------------------------------------------------------------------------
# 单次 LLM 规划调用（C2 详细规划）
# ---------------------------------------------------------------------------

def _make_schema_hint() -> str:
    return json.dumps({
        "doc_type": "string",
        "sections": [{
            "seq_ids": [0, 1],
            "title": "string",
            "content_description": "string",
            "enhance_guidance": "string",
            "rationale": "string",
            "split_at": [3],
        }],
    }, ensure_ascii=False)


def _call_c2(
    outline: str,
    doc_name: str,
    total_sections: int,
    seq_id_min: int,
    seq_id_max: int,
    chunk_min_chars: int,
    chunk_max_chars: int,
    provider: Any,
    system_prompt: str,
    schema_hint: str,
    output_reserve: int,
) -> dict[str, Any]:
    """执行一次 C2 详细规划 LLM 调用。"""
    user_prompt = render_prompt("section_plan.j2", {
        "doc_name": doc_name or "未命名文档",
        "outline": outline,
        "total_sections": total_sections,
        "chunk_min_chars": chunk_min_chars,
        "chunk_max_chars": chunk_max_chars,
        "seq_id_min": seq_id_min,
        "seq_id_max": seq_id_max,
    })
    response = provider.generate_json(
        prompt=user_prompt,
        system_prompt=system_prompt,
        schema_hint=schema_hint,
        max_tokens=output_reserve,
    )
    if not isinstance(response, dict):
        raise ValueError(f"C2 response type={type(response)}")
    return response


def _run_single_call(
    title_items: list[TitleItem],
    body_map: dict[int, BodyItem],
    preamble: BodyItem | None,
    available_input_tokens: int,
    token_counter: TokenCounter,
    provider: Any,
    doc_name: str,
    system_prompt: str,
    schema_hint: str,
    chunk_min_chars: int,
    chunk_max_chars: int,
) -> tuple[list[dict[str, Any]], str, int]:
    """Phase C - 单次 LLM 调用（预算充足时）。最多重试 3 次。"""
    n = len(title_items)
    max_seq_id = n - 1
    output_reserve = max(1200, n * 200)

    outline = _build_outline(
        title_items, body_map, preamble, inline_max_chars=_INLINE_MAX_CHARS,
    )

    doc_type = "unknown"
    fixed_sections: list[dict[str, Any]] | None = None
    llm_calls = 0
    extra_suffix = ""

    for attempt in range(3):
        try:
            llm_calls += 1
            outline_with_hint = outline + extra_suffix
            response = _call_c2(
                outline=outline_with_hint,
                doc_name=doc_name,
                total_sections=n,
                seq_id_min=0,
                seq_id_max=max_seq_id,
                chunk_min_chars=chunk_min_chars,
                chunk_max_chars=chunk_max_chars,
                provider=provider,
                system_prompt=system_prompt,
                schema_hint=schema_hint,
                output_reserve=output_reserve,
            )
            doc_type = str(response.get("doc_type", "unknown") or "unknown").strip() or "unknown"
            raw_sections = response.get("sections")
            if not isinstance(raw_sections, list):
                raise ValueError(f"sections is not a list: {type(raw_sections)}")

            fixed, has_fatal = _validate_and_fix(raw_sections, max_seq_id)
            if not has_fatal and fixed:
                fixed_sections = fixed
                break
            logger.warning(f"[SectionPlanStage] validation issue, attempt={attempt + 1}")

        except Exception as exc:
            logger.warning(f"[SectionPlanStage] LLM call failed: {exc}, attempt={attempt + 1}")

        if attempt == 0:
            extra_suffix = (
                "\n\n【重要】seq_ids 必须是整数列表；"
                f"所有 sections 的 seq_ids 并集必须恰好覆盖 0 到 {max_seq_id} 的全部整数，且每组内部必须连续。"
            )
        elif attempt == 1:
            extra_suffix = (
                f"\n\n【强制要求】必须返回合法 JSON。sections 必须覆盖 seq_id 0~{max_seq_id} 全部，"
                "每组 seq_ids 内值连续无跳跃。"
            )

    if fixed_sections is None:
        return [], doc_type, llm_calls

    return fixed_sections, doc_type, llm_calls


# ---------------------------------------------------------------------------
# 两级规划：C1 骨架 + 并发 C2 章节
# ---------------------------------------------------------------------------

def _validate_c1_chapters(
    raw_chapters: list[Any],
    max_seq_id: int,
) -> list[dict[str, Any]] | None:
    """校验并修复 C1 章节组输出。返回 None 表示结果无法使用。"""
    if not isinstance(raw_chapters, list) or not raw_chapters:
        return None

    valid_range = set(range(max_seq_id + 1))
    used: set[int] = set()
    chapters: list[dict[str, Any]] = []

    for ch in raw_chapters:
        if not isinstance(ch, dict):
            continue
        seq_ids_raw = ch.get("seq_ids", [])
        if not isinstance(seq_ids_raw, list):
            continue

        valid_ids = sorted(
            int(i) for i in seq_ids_raw
            if isinstance(i, (int, float)) and int(i) in valid_range and int(i) not in used
        )
        if not valid_ids:
            continue

        # 强制连续：取首尾之间的全部 id（跳跃情况下取连续范围）
        contiguous = [i for i in range(valid_ids[0], valid_ids[-1] + 1)
                      if i in valid_range and i not in used]
        if not contiguous:
            continue

        chapters.append({"seq_ids": contiguous, "label": str(ch.get("label", ""))})
        used.update(contiguous)

    # 补全缺漏（每个缺漏 id 独立成组）
    missing = valid_range - used
    for sid in sorted(missing):
        chapters.append({"seq_ids": [sid], "label": f"章节{sid:02d}"})
        used.add(sid)

    # 按首 seq_id 排序，保证顺序
    chapters.sort(key=lambda c: c["seq_ids"][0])
    return chapters if chapters else None


def _run_c2_chapter(
    chapter_seq_ids: list[int],
    title_items: list[TitleItem],
    body_map: dict[int, BodyItem],
    preamble: BodyItem | None,
    available_input_tokens: int,
    provider: Any,
    token_counter: TokenCounter,
    doc_name: str,
    doc_type: str,
    system_prompt: str,
    schema_hint: str,
    chunk_min_chars: int,
    chunk_max_chars: int,
) -> tuple[list[dict[str, Any]], int]:
    """对一个章节组执行 C2 详细规划（全局 seq_id 坐标）。

    输入 chapter_seq_ids 为全局 seq_id 列表（连续整数）。
    LLM 输入/输出均使用全局 seq_id，无需本地→全局 remapping。
    若章节大纲超出上下文，退化为每节独立成节（不再递归滑动窗口）。
    """
    n = len(chapter_seq_ids)
    if n == 0:
        return [], 0

    global_offset = chapter_seq_ids[0]
    max_global_id = chapter_seq_ids[-1]

    # 提取本章节的 title_items 和 body_map（保持全局 key）
    local_titles = [title_items[i] for i in chapter_seq_ids if i < len(title_items)]
    local_body_map = {
        local_idx: body_map[global_idx]
        for local_idx, global_idx in enumerate(chapter_seq_ids)
        if global_idx in body_map
    }
    local_preamble = preamble if global_offset == 0 else None

    outline = _build_outline(
        local_titles, local_body_map, local_preamble,
        inline_max_chars=_INLINE_MAX_CHARS,
        title_offset=global_offset,
    )
    output_reserve = max(800, n * 200)
    outline_tokens = token_counter.count(outline)
    sys_tokens = token_counter.count(system_prompt)

    if outline_tokens + sys_tokens + output_reserve > available_input_tokens:
        # 超出上下文：退化为每节独立成节
        logger.warning(
            f"[SectionPlanStage] C2 chapter {chapter_seq_ids[0]}~{chapter_seq_ids[-1]} "
            f"outline too long ({outline_tokens} tokens), fallback to standalone"
        )
        return [
            {
                "seq_ids": [chapter_seq_ids[i]],
                "title": (title_items[chapter_seq_ids[i]].title_text
                          if chapter_seq_ids[i] < len(title_items) else f"章节{i}"),
                "content_description": "",
                "enhance_guidance": "",
                "rationale": "章节过长，独立成节",
            }
            for i in range(n)
        ], 0

    try:
        response = _call_c2(
            outline=outline,
            doc_name=doc_name,
            total_sections=n,
            seq_id_min=global_offset,
            seq_id_max=max_global_id,
            chunk_min_chars=chunk_min_chars,
            chunk_max_chars=chunk_max_chars,
            provider=provider,
            system_prompt=system_prompt,
            schema_hint=schema_hint,
            output_reserve=output_reserve,
        )

        raw_sections = response.get("sections")
        if not isinstance(raw_sections, list):
            raise ValueError("C2 sections not a list")

        # 校验（全局 seq_id 坐标：min=global_offset, max=max_global_id）
        # _validate_and_fix 以 max_seq_id 为上界，但我们的全局 id 从 global_offset 开始
        # 通过临时平移到 [0, n-1] 校验，再平移回去
        shifted = []
        for sec in raw_sections:
            shifted_ids = [i - global_offset for i in sec.get("seq_ids", [])]
            shifted.append({**sec, "seq_ids": shifted_ids})

        fixed_shifted, has_fatal = _validate_and_fix(shifted, n - 1)
        if has_fatal or not fixed_shifted:
            raise ValueError("C2 validation fatal")

        # 平移回全局坐标
        fixed_global = []
        for sec in fixed_shifted:
            global_ids = [i + global_offset for i in sec["seq_ids"]]
            fixed_global.append({**sec, "seq_ids": global_ids})

        return fixed_global, 1

    except Exception as exc:
        logger.warning(
            f"[SectionPlanStage] C2 chapter {chapter_seq_ids[0]}~{chapter_seq_ids[-1]} "
            f"failed: {exc}, fallback to standalone"
        )
        return [
            {
                "seq_ids": [chapter_seq_ids[i]],
                "title": (title_items[chapter_seq_ids[i]].title_text
                          if chapter_seq_ids[i] < len(title_items) else f"章节{i}"),
                "content_description": "",
                "enhance_guidance": "",
                "rationale": "C2调用失败，独立成节",
            }
            for i in range(n)
        ], 1


def _run_two_level_planning(
    title_items: list[TitleItem],
    body_map: dict[int, BodyItem],
    preamble: BodyItem | None,
    available_input_tokens: int,
    token_counter: TokenCounter,
    provider: Any,
    doc_name: str,
    system_prompt: str,
    schema_hint: str,
    chunk_min_chars: int,
    chunk_max_chars: int,
    max_c2_workers: int = 4,
) -> tuple[list[dict[str, Any]], str, int]:
    """两级规划：C1 骨架 → 并发 C2 章节规划。

    C1: 用骨架大纲（标题 + 字符数，无正文内容）换取全局文档感知，输出章节分组。
    C2: 对每个章节组并发执行详细规划，最后合并。
    任一 C1 失败 → 抛出异常（由调用方退化到滑动窗口）。
    """
    n = len(title_items)
    max_seq_id = n - 1

    # C1: 骨架规划
    skeleton = _build_skeleton(title_items, body_map, preamble)
    c1_schema = json.dumps({
        "doc_type": "string",
        "chapters": [{"seq_ids": [0, 1, 2], "label": "string"}],
    }, ensure_ascii=False)

    c1_user = (
        f"文档《{doc_name or '未命名文档'}》共 {n} 个标题节（[00]~[{max_seq_id:02d}]），"
        "请将标题序列划分为若干语义相关的连续章节组：\n\n"
        f"{skeleton}\n\n"
        "返回 JSON：\n"
        '{"doc_type": "标准|论文|手册|法规|报告|其他", '
        '"chapters": [{"seq_ids": [0, 1, 2], "label": "章节组标签"}]}\n'
        f"覆盖约束：所有 seq_ids 并集 = 0~{max_seq_id} 全集，每组内必须连续。"
    )
    c1_output_reserve = max(600, n * 30)

    c1_response = provider.generate_json(
        prompt=c1_user,
        system_prompt=_C1_SYSTEM_PROMPT,
        schema_hint=c1_schema,
        max_tokens=c1_output_reserve,
    )
    if not isinstance(c1_response, dict):
        raise ValueError(f"C1 response type={type(c1_response)}")

    doc_type = str(c1_response.get("doc_type", "unknown") or "unknown").strip() or "unknown"
    raw_chapters = c1_response.get("chapters")
    if not isinstance(raw_chapters, list):
        raise ValueError("C1: chapters not a list")

    chapters = _validate_c1_chapters(raw_chapters, max_seq_id)
    if not chapters:
        raise ValueError("C1: validation failed")

    logger.debug(
        f"[SectionPlanStage] C1 done: doc_type={doc_type}, "
        f"{len(chapters)} chapter groups for {n} titles"
    )

    # C2: 并发章节规划
    llm_calls = 1  # C1

    def _process_chapter(ch: dict[str, Any]) -> list[dict[str, Any]]:
        secs, calls = _run_c2_chapter(
            chapter_seq_ids=ch["seq_ids"],
            title_items=title_items,
            body_map=body_map,
            preamble=preamble,
            available_input_tokens=available_input_tokens,
            provider=provider,
            token_counter=token_counter,
            doc_name=doc_name,
            doc_type=doc_type,
            system_prompt=system_prompt,
            schema_hint=schema_hint,
            chunk_min_chars=chunk_min_chars,
            chunk_max_chars=chunk_max_chars,
        )
        return secs, calls

    # 按 chapters 顺序收集结果（并发执行，结果按原顺序合并）
    chapter_results: dict[int, tuple[list[dict], int]] = {}

    with ThreadPoolExecutor(max_workers=max_c2_workers) as executor:
        future_to_idx = {
            executor.submit(_process_chapter, ch): ch_idx
            for ch_idx, ch in enumerate(chapters)
        }
        for future in as_completed(future_to_idx):
            ch_idx = future_to_idx[future]
            try:
                secs, calls = future.result()
                chapter_results[ch_idx] = (secs, calls)
            except Exception as exc:
                logger.warning(f"[SectionPlanStage] C2 chapter {ch_idx} exception: {exc}")
                ch = chapters[ch_idx]
                fallback_secs = [
                    {
                        "seq_ids": [sid],
                        "title": (title_items[sid].title_text if sid < len(title_items) else f"章节{sid}"),
                        "content_description": "",
                        "enhance_guidance": "",
                        "rationale": "并发异常，独立成节",
                    }
                    for sid in ch["seq_ids"]
                ]
                chapter_results[ch_idx] = (fallback_secs, 0)

    # 按原顺序合并
    all_sections: list[dict[str, Any]] = []
    for ch_idx in range(len(chapters)):
        secs, calls = chapter_results.get(ch_idx, ([], 0))
        all_sections.extend(secs)
        llm_calls += calls

    return all_sections, doc_type, llm_calls


# ---------------------------------------------------------------------------
# 滑动窗口（超长文档兜底）
# ---------------------------------------------------------------------------

def _partition_windows(
    title_items: list[TitleItem],
    body_map: dict[int, BodyItem],
    preamble: BodyItem | None,
    token_counter: TokenCounter,
    window_input_budget: int,
    inline_max_chars: int = _INLINE_MAX_CHARS,
    overlap: int = 2,
) -> list[tuple[list[TitleItem], dict[int, BodyItem], BodyItem | None, int]]:
    """将标题序列切分为滑动窗口，各窗口 token 消耗在预算内。"""
    windows: list[tuple[list[TitleItem], dict[int, BodyItem], BodyItem | None, int]] = []
    n = len(title_items)
    start = 0

    while start < n:
        accumulated = 0
        end = start
        while end < n:
            t = title_items[end]
            t_tokens = token_counter.count(t.title_text) + 20
            body = body_map.get(end)
            b_tokens = 0
            if body and body.char_count > 0:
                if body.char_count <= inline_max_chars:
                    b_tokens = token_counter.count(body.body_text) + 10
                else:
                    # block list: 估计每块约 20 tokens
                    b_tokens = len(body.block_items) * 20 + 10
            needed = t_tokens + b_tokens
            if accumulated + needed > window_input_budget and end > start:
                break
            accumulated += needed
            end += 1

        if end == start:
            end = start + 1

        win_titles = title_items[start:end]
        win_body_map = {
            local_idx: body_map[global_idx]
            for local_idx, global_idx in enumerate(range(start, end))
            if global_idx in body_map
        }
        win_preamble = preamble if start == 0 else None
        windows.append((win_titles, win_body_map, win_preamble, start))

        next_start = end - min(overlap, end - start - 1)
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return windows


def _run_with_sliding_window(
    title_items: list[TitleItem],
    body_map: dict[int, BodyItem],
    preamble: BodyItem | None,
    available_input_tokens: int,
    token_counter: TokenCounter,
    provider: Any,
    doc_name: str,
    system_prompt: str,
    schema_hint: str,
    chunk_min_chars: int,
    chunk_max_chars: int,
) -> tuple[list[dict[str, Any]], str, int]:
    """滑动窗口规划（两级规划失败后的兜底，overlap=2 保证边界语义一致）。"""
    n = len(title_items)
    output_reserve_per_win = max(800, min(n, 15) * 200)
    window_input_budget = available_input_tokens - output_reserve_per_win

    windows = _partition_windows(
        title_items, body_map, preamble, token_counter, window_input_budget,
    )
    logger.debug(f"[SectionPlanStage] sliding window: {len(windows)} windows for {n} titles")

    all_sections: list[dict[str, Any]] = []
    overlap_global_ids: set[int] = set()
    doc_type = "unknown"
    llm_calls = 0

    for win_idx, (win_titles, win_body_map, win_preamble, global_offset) in enumerate(windows):
        local_n = len(win_titles)
        outline = _build_outline(
            win_titles, win_body_map, win_preamble,
            inline_max_chars=_INLINE_MAX_CHARS,
            title_offset=global_offset,
        )
        output_reserve = max(800, local_n * 200)

        win_sections: list[dict[str, Any]] = []
        for attempt in range(2):
            try:
                llm_calls += 1
                response = _call_c2(
                    outline=outline,
                    doc_name=doc_name,
                    total_sections=local_n,
                    seq_id_min=global_offset,
                    seq_id_max=global_offset + local_n - 1,
                    chunk_min_chars=chunk_min_chars,
                    chunk_max_chars=chunk_max_chars,
                    provider=provider,
                    system_prompt=system_prompt,
                    schema_hint=schema_hint,
                    output_reserve=output_reserve,
                )
                if win_idx == 0:
                    doc_type = str(response.get("doc_type", "unknown") or "unknown").strip() or "unknown"

                raw = response.get("sections")
                if not isinstance(raw, list):
                    raise ValueError("sections not a list")

                # 平移到 [0, local_n-1] 校验，再平移回全局
                shifted = [{**s, "seq_ids": [i - global_offset for i in s.get("seq_ids", [])]} for s in raw]
                fixed_shifted, fatal = _validate_and_fix(shifted, local_n - 1)
                if not fatal and fixed_shifted:
                    win_sections = [
                        {**s, "seq_ids": [i + global_offset for i in s["seq_ids"]]}
                        for s in fixed_shifted
                    ]
                    break
            except Exception as exc:
                logger.warning(f"[SectionPlanStage] window {win_idx} attempt {attempt}: {exc}")

        if not win_sections:
            for local_idx, t in enumerate(win_titles):
                win_sections.append({
                    "seq_ids": [global_offset + local_idx],
                    "title": t.title_text,
                    "content_description": "",
                    "enhance_guidance": "",
                    "rationale": "窗口fallback",
                })

        # 过滤重叠（已被上一窗口规划的部分，以后一窗口结果为准）
        if overlap_global_ids:
            win_sections = [
                s for s in win_sections
                if not all(i in overlap_global_ids for i in s.get("seq_ids", []))
            ]
        all_sections.extend(win_sections)

        win_global_ids = list(range(global_offset, global_offset + len(win_titles)))
        overlap_global_ids = set(win_global_ids[-2:]) if len(win_global_ids) >= 2 else set(win_global_ids)

    return all_sections, doc_type, llm_calls


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_section_plan_stage(
    *,
    items: list[TitleItem | BodyItem],
    provider: Any,
    token_counter: TokenCounter,
    available_input_tokens: int,
    enhance_cfg: dict[str, Any],
    language: str = "zh",
    doc_name: str = "",
) -> tuple[list[EnhanceUnit], SectionPlanResult]:
    """Hybrid Chunk Planner 主入口。

    路由逻辑：
      1. 无标题 → 直接 fallback（整文档一节）
      2. 全量 C2 大纲 token 预算充足 → 单次 LLM 调用
      3. 预算不足 → C1 骨架 + 并发 C2 章节；C1 失败 → 退化到滑动窗口
      4. 以上均失败 → 每节独立成节（fallback）
    """
    chunk_max_chars = int(enhance_cfg.get("chunk_max_chars") or _DEFAULT_CHUNK_MAX_CHARS)
    chunk_min_chars = int(enhance_cfg.get("chunk_min_chars") or _DEFAULT_CHUNK_MIN_CHARS)

    title_items, body_map, preamble = _parse_items(items)
    original_section_count = len(title_items)

    # ── 无标题：单节 fallback ───────────────────────────────────────────────
    if not title_items:
        logger.info("[SectionPlanStage] no title items found, single-unit fallback")
        units = _fallback_each_title_standalone([], body_map, preamble, token_counter)
        return units, SectionPlanResult(
            strategy="fallback",
            doc_type="unknown",
            fragmentation_rate=0.0,
            original_section_count=0,
            enhance_unit_count=len(units),
            llm_calls=0,
            fallback=True,
            fallback_reason="no_titles",
        )

    n = len(title_items)
    max_seq_id = n - 1
    system_prompt = render_prompt("section_plan_sys.j2", {})
    schema_hint = _make_schema_hint()

    # ── 估算全量 C2 大纲 token 开销 ─────────────────────────────────────────
    full_outline = _build_outline(title_items, body_map, preamble, inline_max_chars=_INLINE_MAX_CHARS)
    sys_tokens = token_counter.count(system_prompt)
    outline_tokens = token_counter.count(full_outline)
    output_reserve = max(1200, n * 200)
    total_needed = sys_tokens + outline_tokens + output_reserve

    logger.debug(
        f"[SectionPlanStage] token budget: sys={sys_tokens} outline={outline_tokens} "
        f"reserve={output_reserve} total={total_needed} available={available_input_tokens} "
        f"chunk_min={chunk_min_chars} chunk_max={chunk_max_chars}"
    )

    all_sections: list[dict[str, Any]] = []
    doc_type = "unknown"
    llm_calls = 0
    strategy = "llm_assisted"
    fallback = False
    fallback_reason = ""

    if total_needed <= available_input_tokens:
        # ── Path 1: 单次 C2 调用 ──────────────────────────────────────────
        all_sections, doc_type, llm_calls = _run_single_call(
            title_items=title_items,
            body_map=body_map,
            preamble=preamble,
            available_input_tokens=available_input_tokens,
            token_counter=token_counter,
            provider=provider,
            doc_name=doc_name,
            system_prompt=system_prompt,
            schema_hint=schema_hint,
            chunk_min_chars=chunk_min_chars,
            chunk_max_chars=chunk_max_chars,
        )

        if not all_sections:
            logger.error("[SectionPlanStage] single call failed after retries, fallback to standalone")
            units = _fallback_each_title_standalone(title_items, body_map, preamble, token_counter)
            return units, SectionPlanResult(
                strategy="fallback",
                doc_type=doc_type,
                fragmentation_rate=0.0,
                original_section_count=original_section_count,
                enhance_unit_count=len(units),
                llm_calls=llm_calls,
                fallback=True,
                fallback_reason="llm_failed",
            )

    else:
        # ── Path 2: 两级规划（C1 骨架 + 并发 C2）───────────────────────────
        logger.warning(
            f"[SectionPlanStage] token budget exceeded ({total_needed} > {available_input_tokens}), "
            "trying two-level planning (C1+C2)"
        )
        try:
            all_sections, doc_type, llm_calls = _run_two_level_planning(
                title_items=title_items,
                body_map=body_map,
                preamble=preamble,
                available_input_tokens=available_input_tokens,
                token_counter=token_counter,
                provider=provider,
                doc_name=doc_name,
                system_prompt=system_prompt,
                schema_hint=schema_hint,
                chunk_min_chars=chunk_min_chars,
                chunk_max_chars=chunk_max_chars,
            )
        except Exception as exc:
            logger.warning(
                f"[SectionPlanStage] C1 failed ({exc}), fallback to sliding window"
            )
            all_sections, doc_type, llm_calls = _run_with_sliding_window(
                title_items=title_items,
                body_map=body_map,
                preamble=preamble,
                available_input_tokens=available_input_tokens,
                token_counter=token_counter,
                provider=provider,
                doc_name=doc_name,
                system_prompt=system_prompt,
                schema_hint=schema_hint,
                chunk_min_chars=chunk_min_chars,
                chunk_max_chars=chunk_max_chars,
            )

        # 全局校验
        all_fixed, has_fatal = _validate_and_fix(all_sections, max_seq_id)
        if has_fatal or not all_fixed:
            logger.error("[SectionPlanStage] global validation failed, fallback to standalone")
            units = _fallback_each_title_standalone(title_items, body_map, preamble, token_counter)
            return units, SectionPlanResult(
                strategy="fallback",
                doc_type=doc_type,
                fragmentation_rate=0.0,
                original_section_count=original_section_count,
                enhance_unit_count=len(units),
                llm_calls=llm_calls,
                fallback=True,
                fallback_reason="sliding_window_failed",
            )
        all_sections = all_fixed

    # ── Phase E: 组装 EnhanceUnit ──────────────────────────────────────────
    units = build_enhance_units(all_sections, title_items, body_map, preamble, token_counter)
    # 兜底：段落对齐分割仍超长的单元
    units = _split_oversized_units(units, chunk_max_chars, token_counter=token_counter)

    frag_rate = round(1.0 - len(units) / original_section_count, 4) if original_section_count > 0 else 0.0
    logger.info(
        f"[SectionPlanStage] success: {original_section_count} titles → "
        f"{len(units)} enhance units, doc_type={doc_type}, llm_calls={llm_calls}"
    )

    return units, SectionPlanResult(
        strategy=strategy,
        doc_type=doc_type,
        fragmentation_rate=frag_rate,
        original_section_count=original_section_count,
        enhance_unit_count=len(units),
        llm_calls=llm_calls,
        fallback=fallback,
        fallback_reason=fallback_reason,
    )
