"""Stage 2: LLM Section Planner — 替换旧 SectionPlanStage（树形结构）。

输入：Block Indexer 产出的 list[TitleItem | BodyItem]（flat 物理顺序）
输出：list[EnhanceUnit]（flat，无树结构）+ SectionPlanResult

核心设计：
  - 不传 level_hint 给 LLM（level 不可靠）
  - LLM 以 flat table 形式接收，语义驱动分组，无硬性字数/数量约束
  - 三层校验：存在性 → 连续性 → 覆盖性（含自动修复）
  - section_id 在校验通过后才分配（s001, s002, …）
  - block_refs 后置构建（物理顺序有序 union，100% 合法）
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from loguru import logger

from mineru.custom.enhance_mvp.prompts import render_prompt
from mineru.custom.enhance_mvp.schema import EnhanceUnit
from mineru.custom.enhance_mvp.stages_block_indexer import BodyItem, TitleItem
from mineru.custom.enhance_mvp.utils import TokenCounter


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
# LLM 输入构建
# ---------------------------------------------------------------------------

def _build_llm_input(
    title_items: list[TitleItem],
    body_map: dict[int, BodyItem],
    preamble: BodyItem | None,
) -> str:
    """构建 type|text|chars|refs 四列格式的 LLM 输入。

    格式（含表头）：
      type      | "text/preview"   | chars | refs
      body      | "..."            | N     | p,b,type; p,b,type
      title[NN] | "标题文本"        | N     | p,b,title

    - body 行无序号，位置隐含与前一 title 的归属关系
    - title[NN] 中 NN 与输出 seq_ids 直接对应（从 00 起两位补零）
    - chars：title = len(title_text)，body = char_count
    - refs：page,block,type 三元组，多个用 '; ' 分隔，无括号
    """
    def _format_refs(refs: list[list]) -> str:
        """将 block_refs 转为 'page,block,type; ...' 格式。"""
        return "; ".join(
            f"{r[0]},{r[1]},{r[2]}"
            for r in refs
            if isinstance(r, list) and len(r) >= 3
        )

    lines: list[str] = [
        'type      | "text/preview"   | chars | refs',
    ]

    if preamble and (preamble.char_count > 0 or preamble.block_refs):
        refs_str = _format_refs(preamble.block_refs)
        lines.append(f'body      | "{preamble.preview}" | {preamble.char_count} | {refs_str}')

    for title_idx, t in enumerate(title_items):
        refs_str = _format_refs([t.block_ref])
        chars = len(t.title_text)
        lines.append(f'title[{title_idx:02d}] | "{t.title_text}" | {chars} | {refs_str}')

        body = body_map.get(title_idx)
        if body and (body.char_count > 0 or body.block_refs):
            refs_str = _format_refs(body.block_refs)
            lines.append(f'body      | "{body.preview}" | {body.char_count} | {refs_str}')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 三层校验 + 自动修复
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
    fixed_sections 中每个元素保证: {seq_ids: list[int], title: str, rationale: str}
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

        # Layer 1: 过滤非法 seq_id（不在范围内或已使用）
        valid_ids = [
            int(sid) for sid in seq_ids_raw
            if isinstance(sid, (int, float)) and int(sid) in valid_range and int(sid) not in used_ids
        ]
        if not valid_ids:
            continue

        # Layer 2: 连续性修复（非连续则拆分成多个连续组）
        valid_ids_sorted = sorted(valid_ids)
        groups: list[list[int]] = [[valid_ids_sorted[0]]]
        for sid in valid_ids_sorted[1:]:
            if sid == groups[-1][-1] + 1:
                groups[-1].append(sid)
            else:
                groups.append([sid])

        title = str(item.get("title", "")).strip() or "章节单元"
        rationale = str(item.get("rationale", "")).strip() or "语义分组"

        for grp in groups:
            if not grp:
                continue
            fixed.append({
                "seq_ids": grp,
                "title": title,
                "rationale": rationale,
            })
            used_ids.update(grp)

    # Layer 3: 覆盖性修复（补全遗漏的 seq_ids，每个遗漏 seq_id 独立成节，避免级联追加）
    missing = valid_range - used_ids
    if missing:
        logger.debug(f"[SectionPlanStage] coverage repair: missing seq_ids={sorted(missing)}")
        for sid in sorted(missing):
            fixed.append({"seq_ids": [sid], "title": f"章节{sid:02d}", "rationale": "自动补全"})
            used_ids.add(sid)

    return fixed, False


# ---------------------------------------------------------------------------
# EnhanceUnit 构建
# ---------------------------------------------------------------------------

def _compute_page_range(block_refs: list[list]) -> list[int]:
    """从三元列表引用 [page_no, block_no, type] 中提取页码范围（1-based）。"""
    pages = [r[0] for r in block_refs if isinstance(r, list) and len(r) >= 1]
    if not pages:
        return [-1, -1]
    return [min(pages), max(pages)]


def _parse_items(
    items: list[TitleItem | BodyItem],
) -> tuple[list[TitleItem], dict[int, BodyItem], BodyItem | None]:
    """从 BlockIndexer 输出的 flat list 中解析出 title/body 的位置关联。

    - title_items[i] 对应文档中第 i 个 title（0-based 枚举位置）
    - body_map[i] 为紧跟 title_items[i] 之后的 BodyItem（若有内容才存在）
    - preamble：首个 TitleItem 之前出现的 BodyItem（若有）

    这里不使用任何存储在 item 中的 id 字段，完全依赖列表顺序。
    """
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


def build_enhance_units(
    validated_sections: list[dict[str, Any]],
    title_items: list[TitleItem],
    body_map: dict[int, BodyItem],
    preamble: BodyItem | None,
) -> list[EnhanceUnit]:
    """根据校验通过的 sections 规划，构建 EnhanceUnit 列表。

    section_id 在此处分配（s001, s002, …）。
    block_refs 为顺序 union（TitleItem.block_ref + BodyItem.block_refs），
    均为三元列表 [page_no_1based, block_no_1based, type_str]。
    """
    units: list[EnhanceUnit] = []

    for i, section in enumerate(validated_sections):
        section_id = f"s{i + 1:03d}"
        seq_ids: list[int] = sorted(section["seq_ids"])
        block_refs: list[list] = []

        # preamble 归入覆盖 seq_id=0 的首个 section（若有 preamble）
        if seq_ids and seq_ids[0] == 0 and preamble is not None:
            block_refs.extend(preamble.block_refs)

        body_text_parts: list[str] = []
        for idx in seq_ids:
            t = title_items[idx] if idx < len(title_items) else None
            if t is None:
                continue
            # title block ref（已是三元列表）
            block_refs.append(t.block_ref)
            # body blocks
            body = body_map.get(idx)
            if body:
                block_refs.extend(body.block_refs)
                if body.body_text.strip():
                    body_text_parts.append(body.body_text.strip())

        # title 使用 LLM 给出的 title（可能是合并单元标题）
        unit_title = section.get("title", "")
        if not unit_title and seq_ids:
            # fallback：使用首个 TitleItem 的原始标题
            first_idx = seq_ids[0]
            first_t = title_items[first_idx] if first_idx < len(title_items) else None
            unit_title = first_t.title_text if first_t else f"章节{seq_ids[0]:02d}"

        # 插入锚点：永远使用覆盖范围内第一个原始标题（用于 ComposeStage 定位到 markdown 标题行）
        anchor_title = ""
        if seq_ids:
            first_idx = seq_ids[0]
            first_t = title_items[first_idx] if first_idx < len(title_items) else None
            anchor_title = (first_t.title_text if first_t else "").strip()

        units.append(EnhanceUnit(
            section_id=section_id,
            title=unit_title,
            anchor_title=anchor_title,
            seq_ids=seq_ids,
            body_text="\n".join(body_text_parts),
            block_refs=block_refs,
            page_range=_compute_page_range(block_refs),
        ))

    return units


# ---------------------------------------------------------------------------
# Fallback：每个 TitleItem 独立成节
# ---------------------------------------------------------------------------

def _fallback_each_title_standalone(
    title_items: list[TitleItem],
    body_map: dict[int, BodyItem],
    preamble: BodyItem | None,
) -> list[EnhanceUnit]:
    """兜底：每个 TitleItem 独立成一个 EnhanceUnit。"""
    if not title_items:
        # 文档无 title：整个文档作为一个 unit
        block_refs: list[list] = []
        body_text = ""
        if preamble:
            block_refs.extend(preamble.block_refs)
            body_text = preamble.body_text.strip()
        return [EnhanceUnit(
            section_id="s001",
            title="全文内容",
            anchor_title="",
            seq_ids=[],
            body_text=body_text,
            block_refs=block_refs,
            page_range=_compute_page_range(block_refs),
        )]

    units: list[EnhanceUnit] = []
    for i, t in enumerate(title_items):
        section_id = f"s{i + 1:03d}"
        block_refs = [t.block_ref]  # 已是三元列表
        body = body_map.get(i)
        if body:
            block_refs.extend(body.block_refs)
        # preamble 归入首个 unit
        if i == 0 and preamble:
            block_refs = list(preamble.block_refs) + block_refs
        units.append(EnhanceUnit(
            section_id=section_id,
            title=t.title_text,
            anchor_title=t.title_text,
            seq_ids=[i],
            body_text=body.body_text.strip() if body else "",
            block_refs=block_refs,
            page_range=_compute_page_range(block_refs),
        ))
    return units


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
    """LLM Section Planner 主入口。

    参数:
        items:                  Block Indexer 产出的 TitleItem/BodyItem 列表
        provider:               LLM provider（支持 generate_json）
        token_counter:          token 计数器
        available_input_tokens: 安全余量后的输入 token 上限
        enhance_cfg:            enhance 配置 dict
        language:               语言（默认 zh）
        doc_name:               文档名（用于 prompt）

    返回:
        (enhance_units, plan_result)
    """
    # 从 flat list 解析 title/body 位置关联（无需 seq_id / after_seq_id 字段）
    title_items, body_map, preamble = _parse_items(items)

    original_section_count = len(title_items)

    # 无 title：fallback（整个文档为一个单元）
    if not title_items:
        logger.info("[SectionPlanStage] no title items found, single-unit fallback")
        units = _fallback_each_title_standalone([], body_map, preamble)
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

    max_seq_id = len(title_items) - 1  # 0-based 枚举位置的最大值

    # 构建 flat table
    flat_table = _build_llm_input(title_items, body_map, preamble)

    # 渲染 prompt
    system_prompt = render_prompt("section_plan_sys.j2", {})
    user_prompt = render_prompt(
        "section_plan.j2",
        {
            "doc_name": doc_name or "未命名文档",
            "flat_table": flat_table,
            "total_sections": len(title_items),
        },
    )

    # 估算 token
    sys_tokens = token_counter.count(system_prompt)
    user_tokens = token_counter.count(user_prompt)
    output_reserve = max(500, len(title_items) * 20)
    total_needed = sys_tokens + user_tokens + output_reserve

    logger.debug(
        f"[SectionPlanStage] token budget: sys={sys_tokens} user={user_tokens} "
        f"reserve={output_reserve} total={total_needed} available={available_input_tokens}"
    )

    if total_needed > available_input_tokens:
        logger.warning(
            f"[SectionPlanStage] token budget exceeded ({total_needed} > {available_input_tokens}), "
            "fallback to standalone"
        )
        units = _fallback_each_title_standalone(title_items, body_map, preamble)
        return units, SectionPlanResult(
            strategy="fallback",
            doc_type="unknown",
            fragmentation_rate=0.0,
            original_section_count=original_section_count,
            enhance_unit_count=len(units),
            llm_calls=0,
            fallback=True,
            fallback_reason="token_budget_exceeded",
        )

    # schema hint
    schema_hint = json.dumps({
        "doc_type": "string (标准|论文|手册|法规|报告|其他)",
        "sections": [
            {"seq_ids": [0, 1, 2], "title": "string", "rationale": "string"}
        ],
    }, ensure_ascii=False)

    # LLM 调用（最多 3 次）
    llm_calls = 0
    doc_type = "unknown"
    fixed_sections: list[dict[str, Any]] | None = None
    current_user_prompt = user_prompt

    for attempt in range(3):
        try:
            llm_calls += 1
            response = provider.generate_json(
                prompt=current_user_prompt,
                system_prompt=system_prompt,
                schema_hint=schema_hint,
            )

            if not isinstance(response, dict):
                raise ValueError(f"response type={type(response)}")

            doc_type = str(response.get("doc_type", "unknown")).strip() or "unknown"
            raw_sections = response.get("sections")
            if not isinstance(raw_sections, list):
                raise ValueError(f"sections is not a list: {type(raw_sections)}")

            fixed, has_fatal = _validate_and_fix(raw_sections, max_seq_id)

            if not has_fatal and fixed:
                fixed_sections = fixed
                break
            else:
                logger.warning(
                    f"[SectionPlanStage] validation issue, attempt={attempt + 1}"
                )

        except Exception as exc:
            logger.warning(f"[SectionPlanStage] LLM call failed: {exc}, attempt={attempt + 1}")

        # 重试提示强化
        if attempt == 0:
            current_user_prompt = (
                user_prompt
                + "\n\n【重要】seq_ids 必须是整数列表；"
                "所有 sections 的 seq_ids 并集必须恰好覆盖 0 到 "
                f"{max_seq_id} 的全部整数，且每组内部必须连续。"
            )
        elif attempt == 1:
            current_user_prompt = (
                user_prompt
                + f"\n\n【强制要求】必须返回合法 JSON。sections 必须覆盖 seq_id 0~{max_seq_id} 全部，"
                "每组 seq_ids 内值连续无跳跃。"
            )

    if fixed_sections is None:
        logger.error("[SectionPlanStage] failed after retries, fallback to standalone")
        units = _fallback_each_title_standalone(title_items, body_map, preamble)
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

    # 构建 EnhanceUnit
    units = build_enhance_units(fixed_sections, title_items, body_map, preamble)

    logger.info(
        f"[SectionPlanStage] success: {original_section_count} titles → "
        f"{len(units)} enhance units, doc_type={doc_type}, llm_calls={llm_calls}"
    )

    # fragmentation_rate: 原始 title 数 / enhance unit 数（>1 说明有合并）
    frag_rate = round(1.0 - len(units) / original_section_count, 4) if original_section_count > 0 else 0.0

    return units, SectionPlanResult(
        strategy="llm_assisted",
        doc_type=doc_type,
        fragmentation_rate=frag_rate,
        original_section_count=original_section_count,
        enhance_unit_count=len(units),
        llm_calls=llm_calls,
        fallback=False,
    )
