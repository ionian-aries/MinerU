"""Stage 4: SectionEnhanceStage — 对每个 EnhanceUnit 调用 LLM 生成增强内容。

重构后的设计（参见 notes/20）：
  - 输入：list[EnhanceUnit]（flat，无树结构）
  - 每个 unit 直接增强一次（废弃 bottom-up 聚合）
  - Mode A/B/C：装箱并发 + 超长 map-reduce 回退（逻辑不变）
  - 新增 questions 字段（HyDE 思想：每个 section 能回答的典型用户问题）
  - summary：技术密集型，80-150字，含具体数值/标准编号
  - keywords：5-8个，领域专属，禁通用词

废弃：
  - enhance_depth 参数
  - _collect_leaf() 树遍历
  - bottom-up parent 增强循环
  - _build_result_tree() 嵌套输出
  - source_section_ids / merge_type 字段
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from mineru.custom.enhance_mvp.prompts import render_prompt
from mineru.custom.enhance_mvp.schema import EnhanceUnit, validate_section_payload
from mineru.custom.enhance_mvp.utils import TokenCounter, split_text_by_token_budget

# ── system prompt 常量 ──────────────────────────────────────────────
_SECTION_SYSTEM_PROMPT = (
    "你是文档语义增强助手。必须返回JSON对象。"
    "summary 必须自包含且技术密集——禁止'本章节''本节''如前所述'等表述；"
    "必须包含具体数值、参数名称、标准编号。"
    "main_idea 是核心命题，不是 summary 的缩短版。"
    "keywords 必须 5~8 个，领域专属术语，禁止通用词。"
    "questions 是本节能直接回答的 2-3 个典型用户问题。"
)
_BATCH_SYSTEM_PROMPT = (
    "你是文档语义增强助手。必须返回JSON对象。顶层键为section_id。"
    "每个章节的 summary 必须自包含且技术密集——禁止'本章节''本节'等表述；"
    "必须包含具体数值、参数名称、标准编号。"
    "不同章节的 keywords 应体现差异性，5~8个领域专属词，禁止通用词。"
    "questions 是本节能直接回答的 2-3 个典型用户问题。"
)

# questions 字段 schema（供 schema_hint 使用）
_SECTION_SCHEMA_SINGLE = {"summary": "string", "keywords": ["string"], "main_idea": "string", "questions": ["string"]}


def _refs_to_tuples(block_refs: list[dict[str, Any]]) -> list[list[Any]]:
    """block_refs 对象列表 → refs 三元组列表（1-based block_idx）。

    [{"page_idx": 0, "block_idx": 0, "type": "paragraph"}] → [[0, 1, "paragraph"]]
    """
    out: list[list[Any]] = []
    for r in block_refs:
        if isinstance(r, dict):
            out.append([r.get("page_idx", 0), r.get("block_idx", 0) + 1, r.get("type", "paragraph")])
        elif isinstance(r, (list, tuple)):
            out.append(list(r))
    return out


@dataclass
class _EnhanceTask:
    section_id: str      # prompt 输出键（chunk 任务会带 chunk 标记）
    real_section_id: str # 最终结果回填的真实 section_id
    title: str
    text: str
    estimated_tokens: int
    page_range: list[int]
    block_refs: list[dict[str, Any]]
    chunk_index: Optional[int] = None


def _per_section_overhead_tokens(title: str) -> int:
    return 30 + (len(title) // 4 if title else 0)


def _enhance_single_via_prompt(
    *,
    provider: Any,
    language: str,
    title: str,
    section_text: str,
    system_prompt: str,
) -> dict[str, Any]:
    user_prompt = render_prompt(
        "section_summary.j2",
        {
            "language": language,
            "max_words": 320,
            "style": "retrieval_dense",
            "title": title,
            "section_text": section_text,
        },
    )
    schema_hint = json.dumps(_SECTION_SCHEMA_SINGLE, ensure_ascii=False)
    candidate = provider.generate_json(
        prompt=user_prompt,
        system_prompt=system_prompt,
        schema_hint=schema_hint,
    )
    ok, reason = validate_section_payload(candidate)
    if not ok:
        raise ValueError(f"validate_section_payload failed: {reason}")
    return candidate


def _execute_batch(
    *,
    provider: Any,
    language: str,
    tasks: list[_EnhanceTask],
    model_reference: str,
) -> dict[str, dict[str, Any]]:
    """Mode A/B：批量调用 section_batch_summary.j2"""
    sections_block = "\n\n".join(
        f"section_id: {t.section_id}\ntitle: {t.title}\ntext:\n{t.text}"
        for t in tasks
    )
    user_prompt = render_prompt(
        "section_batch_summary.j2",
        {
            "language": language,
            "max_words": 320,
            "style": "retrieval_dense",
            "sections_block": sections_block,
        },
    )
    schema_hint = json.dumps(
        {t.section_id: _SECTION_SCHEMA_SINGLE for t in tasks},
        ensure_ascii=False,
    )
    payload = provider.generate_json(
        prompt=user_prompt,
        system_prompt=_BATCH_SYSTEM_PROMPT,
        schema_hint=schema_hint,
    )
    if not isinstance(payload, dict):
        raise ValueError("batch response is not a dict")

    out: dict[str, dict[str, Any]] = {}
    for t in tasks:
        cand = payload.get(t.section_id)
        if not isinstance(cand, dict):
            raise ValueError(f"batch response missing section_id={t.section_id}")
        ok, reason = validate_section_payload(cand)
        if not ok:
            raise ValueError(f"batch section_id={t.section_id} validate failed: {reason}")
        out[t.section_id] = cand
    return out


def _build_batch_sections_block(tasks: list[_EnhanceTask]) -> str:
    return "\n\n".join(
        f"section_id: {t.section_id}\ntitle: {t.title}\ntext:\n{t.text}"
        for t in tasks
    )


def _estimate_single_request_budget(
    *,
    token_counter: TokenCounter,
    language: str,
    available_input_tokens: int,
) -> int:
    prompt = render_prompt(
        "section_summary.j2",
        {"language": language, "max_words": 320, "style": "retrieval_dense", "title": "", "section_text": ""},
    )
    schema_hint = json.dumps(_SECTION_SCHEMA_SINGLE, ensure_ascii=False)
    fixed = (
        token_counter.count(_SECTION_SYSTEM_PROMPT)
        + token_counter.count(prompt)
        + token_counter.count(schema_hint)
    )
    return max(1, available_input_tokens - fixed - 150)  # 150 output reserve


def _estimate_batch_total_tokens(
    *,
    token_counter: TokenCounter,
    language: str,
    tasks: list[_EnhanceTask],
) -> int:
    sections_block = _build_batch_sections_block(tasks)
    prompt = render_prompt(
        "section_batch_summary.j2",
        {"language": language, "max_words": 320, "style": "retrieval_dense", "sections_block": sections_block},
    )
    schema_hint = json.dumps(
        {t.section_id: _SECTION_SCHEMA_SINGLE for t in tasks}, ensure_ascii=False
    )
    output_reserve = 100 * len(tasks) + 60
    return (
        token_counter.count(_BATCH_SYSTEM_PROMPT)
        + token_counter.count(prompt)
        + token_counter.count(schema_hint)
        + output_reserve
    )


def _schedule_enhance_tasks(
    *,
    provider: Any,
    language: str,
    model_reference: str,
    context_window: int,
    available_input_tokens: int,
    tasks: list[_EnhanceTask],
    token_counter: TokenCounter,
    concurrency: int = 4,
) -> dict[str, dict[str, Any]]:
    """Phase A/B/C：排序 + FFD 装箱 + 并发执行。"""
    concurrency = max(1, min(int(concurrency), 16))

    single_section_budget = _estimate_single_request_budget(
        token_counter=token_counter,
        language=language,
        available_input_tokens=available_input_tokens,
    )

    # Mode C 超长检测：拆 chunk
    direct_tasks: list[_EnhanceTask] = []
    chunk_tasks: list[_EnhanceTask] = []
    chunk_groups: dict[str, list[_EnhanceTask]] = {}
    base_tasks_by_id: dict[str, _EnhanceTask] = {t.section_id: t for t in tasks}

    for t in tasks:
        if t.estimated_tokens > single_section_budget:
            chunks = split_text_by_token_budget(t.text, single_section_budget, token_counter)
            if not chunks:
                direct_tasks.append(t)
                continue
            group: list[_EnhanceTask] = []
            for i, ch in enumerate(chunks):
                chunk_id = f"{t.section_id}__chunk{i:02d}"
                group.append(_EnhanceTask(
                    section_id=chunk_id,
                    real_section_id=t.real_section_id,
                    title=t.title,
                    text=ch,
                    estimated_tokens=token_counter.count(ch),
                    page_range=t.page_range,
                    block_refs=t.block_refs,
                    chunk_index=i,
                ))
            chunk_groups[t.section_id] = group
            chunk_tasks.extend(group)
        else:
            direct_tasks.append(t)

    scheduled_tasks = direct_tasks + chunk_tasks
    if not scheduled_tasks:
        return {}

    # Mode A：全量单次
    total_tokens = _estimate_batch_total_tokens(
        token_counter=token_counter, language=language, tasks=scheduled_tasks
    )
    use_single_batch = total_tokens <= int(available_input_tokens * 0.9)

    # Mode B：FFD 装箱
    sorted_tasks = sorted(scheduled_tasks, key=lambda x: x.estimated_tokens, reverse=True)
    batches: list[list[_EnhanceTask]] = []
    forced_single: list[_EnhanceTask] = []

    if use_single_batch:
        batches = [sorted_tasks]
    else:
        for t in sorted_tasks:
            placed = False
            for idx, batch in enumerate(batches):
                projected = _estimate_batch_total_tokens(
                    token_counter=token_counter, language=language, tasks=batch + [t]
                )
                if projected <= available_input_tokens:
                    batches[idx] = batch + [t]
                    placed = True
                    break
            if not placed:
                single_tokens = _estimate_batch_total_tokens(
                    token_counter=token_counter, language=language, tasks=[t]
                )
                if single_tokens > available_input_tokens:
                    forced_single.append(t)
                else:
                    batches.append([t])

    # 并发执行
    results: dict[str, dict[str, Any]] = {}

    def _run_one_batch(batch_tasks: list[_EnhanceTask]) -> dict[str, dict[str, Any]]:
        tries = 0
        last_exc: Optional[Exception] = None
        while tries < 2:
            tries += 1
            try:
                return _execute_batch(
                    provider=provider, language=language,
                    tasks=batch_tasks, model_reference=model_reference,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        # batch 失败 → 拆成单个调用
        logger.warning(f"[SectionEnhanceStage] batch failed, fallback to single: {last_exc}")
        out_single: dict[str, dict[str, Any]] = {}
        for t in batch_tasks:
            try:
                out_single[t.section_id] = _enhance_single_via_prompt(
                    provider=provider, language=language,
                    title=t.title, section_text=t.text,
                    system_prompt=_SECTION_SYSTEM_PROMPT,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[SectionEnhanceStage] single failed: section_id={t.section_id}: {exc}")
                out_single[t.section_id] = {"__failed__": True}
        return out_single

    if concurrency <= 1 or len(batches) <= 1:
        for b in batches:
            results.update(_run_one_batch(b))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_batch = {executor.submit(_run_one_batch, b): b for b in batches}
            for future in as_completed(future_to_batch):
                batch_tasks = future_to_batch[future]
                try:
                    results.update(future.result())
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[SectionEnhanceStage] batch execution failed: {exc}")
                    for t in batch_tasks:
                        results[t.section_id] = {"__failed__": True}

    for t in forced_single:
        try:
            results[t.section_id] = _enhance_single_via_prompt(
                provider=provider, language=language,
                title=t.title, section_text=t.text,
                system_prompt=_SECTION_SYSTEM_PROMPT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[SectionEnhanceStage] forced single failed: {t.section_id}: {exc}")
            results[t.section_id] = {"__failed__": True}

    # map-reduce 归并：chunk → base
    final_results: dict[str, dict[str, Any]] = {}
    for base_id, base_task in base_tasks_by_id.items():
        base_payload = results.get(base_id)
        if isinstance(base_payload, dict) and not base_payload.get("__failed__"):
            final_results[base_id] = base_payload
            continue

        chunk_group = chunk_groups.get(base_id, [])
        if not chunk_group:
            final_results[base_id] = {"__failed__": True}
            continue

        sorted_chunks = sorted(chunk_group, key=lambda x: x.chunk_index or 0)
        chunk_summaries: list[str] = []
        chunk_main_ideas: list[str] = []
        chunk_failed = False
        for ch in sorted_chunks:
            p = results.get(ch.section_id)
            if not isinstance(p, dict) or p.get("__failed__"):
                chunk_failed = True
                break
            chunk_summaries.append(str(p.get("summary", "")).strip())
            chunk_main_ideas.append(str(p.get("main_idea", "")).strip())

        if chunk_failed:
            final_results[base_id] = {"__failed__": True}
            continue

        merged_text = "\n".join(
            [p for p in (chunk_summaries + chunk_main_ideas) if p.strip()]
        ).strip()
        if not merged_text:
            final_results[base_id] = {"__failed__": True}
            continue

        try:
            final_results[base_id] = _enhance_single_via_prompt(
                provider=provider, language=language,
                title=base_task.title, section_text=merged_text,
                system_prompt=_SECTION_SYSTEM_PROMPT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[SectionEnhanceStage] map-reduce merge failed for {base_id}: {exc}")
            final_results[base_id] = {"__failed__": True}

    return final_results


# ---------------------------------------------------------------------------
# 主入口（flat EnhanceUnit 版本）
# ---------------------------------------------------------------------------

def run_section_enhance_stage(
    *,
    enhance_units: list[EnhanceUnit],
    provider: Any,
    model_reference: str,
    context_window: int,
    available_input_tokens: int,
    concurrency: int = 4,
    language: str = "zh",
    token_counter: TokenCounter | None = None,
) -> list[dict[str, Any]]:
    """SectionEnhanceStage（重构版）：

    - 输入：list[EnhanceUnit]（flat，由 LLM Section Planner 产出）
    - 每个 unit 直接增强一次（无 bottom-up）
    - 输出：flat list[dict]（无 subsections 嵌套）
    """
    if token_counter is None:
        token_counter = TokenCounter.from_model(model_reference)

    # 构建 task 列表
    tasks: list[_EnhanceTask] = []
    for unit in enhance_units:
        tasks.append(_EnhanceTask(
            section_id=unit.section_id,
            real_section_id=unit.section_id,
            title=unit.title,
            text=unit.body_text,
            estimated_tokens=unit.estimated_tokens if unit.estimated_tokens > 0
                             else token_counter.count(unit.body_text),
            page_range=unit.page_range,
            block_refs=unit.block_refs,
        ))

    results_payload = _schedule_enhance_tasks(
        provider=provider,
        language=language,
        model_reference=model_reference,
        context_window=context_window,
        available_input_tokens=available_input_tokens,
        tasks=tasks,
        token_counter=token_counter,
        concurrency=concurrency,
    )

    # 组装输出
    output: list[dict[str, Any]] = []
    for unit in enhance_units:
        payload = results_payload.get(unit.section_id)
        if not isinstance(payload, dict) or payload.get("__failed__"):
            output.append({
                "id": unit.section_id,
                "title": unit.title,
                "anchor_title": getattr(unit, "anchor_title", "") or "",
                "page_range": unit.page_range,
                "refs": _refs_to_tuples(unit.block_refs),
                "status": "failed",
                "error": "enhance_failed",
            })
            continue

        result: dict[str, Any] = {
            "id": unit.section_id,
            "title": unit.title,
            "anchor_title": getattr(unit, "anchor_title", "") or "",
            "page_range": unit.page_range,
            "refs": _refs_to_tuples(unit.block_refs),
            "status": "success",
            "summary": str(payload.get("summary", "")).strip(),
            "keywords": payload.get("keywords", []),
            "main_idea": str(payload.get("main_idea", "")).strip(),
            "questions": payload.get("questions", []),
        }
        output.append(result)

    success_count = sum(1 for r in output if r.get("status") == "success")
    logger.info(
        f"[SectionEnhanceStage] done: total={len(output)} success={success_count} "
        f"failed={len(output) - success_count}"
    )
    return output
