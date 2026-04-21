"""Enhancement pipeline 主编排入口（重构版）。

7 阶段（notes/20）:
  Stage 1: InputStage
  Stage 2: BlockIndexer（新，替换 SectionBuildStage）
  Stage 3: LLM Section Planner（新，替换旧 SectionPlanStage）
  Stage 4: TokenEstimationStage（简化为 flat EnhanceUnit）
  Stage 5: SectionEnhanceStage（简化，废弃 bottom-up）
  Stage 6: DocEnhanceStage（新增 answerable_questions）
  Stage 7: ValidateStage
  Stage 8: ComposeStage
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from mineru.custom.custom_config_loader import CustomConfig
from mineru.custom.enhance_mvp.provider import build_provider_from_custom_config
from mineru.custom.enhance_mvp.stages_block_indexer import run_block_indexer
from mineru.custom.enhance_mvp.stages_compose import run_compose_stage
from mineru.custom.enhance_mvp.stages_doc_enhance import run_doc_enhance_stage
from mineru.custom.enhance_mvp.stages_input import run_input_stage
from mineru.custom.enhance_mvp.stages_section_enhance import run_section_enhance_stage
from mineru.custom.enhance_mvp.stages_section_plan import run_section_plan_stage
from mineru.custom.enhance_mvp.stages_token_estimation import run_token_estimation_stage
from mineru.custom.enhance_mvp.stages_validate import run_validate_stage


def run_enhancement_pipeline(
    *,
    custom_config: CustomConfig,
    pdf_file_name: str,
    process_mode: str,
    middle_json: dict[str, Any],
    content_list_v2: Any,
    raw_markdown: str,
) -> tuple[dict[str, Any], str]:
    # Stage 1: InputStage
    run_input_stage(
        middle_json=middle_json,
        content_list_v2=content_list_v2,
        pdf_file_name=pdf_file_name,
        process_mode=process_mode,
    )

    enhance_cfg = custom_config.enhance or {}
    language = str(enhance_cfg.get("language", "zh") or "zh")
    model = str(enhance_cfg.get("model", "")).strip()
    model_reference = str(enhance_cfg.get("model_reference", "")).strip() or model
    if not model_reference:
        raise ValueError("Enhancement model_reference is empty in custom config.")
    safety_ratio = float(enhance_cfg.get("safety_ratio", 0.85) or 0.85)
    concurrency = int(enhance_cfg.get("concurrency", 4) or 4)

    # Stage 2: Block Indexer（替换 SectionBuildStage）
    items = run_block_indexer(content_list_v2, custom_config)
    logger.debug(f"[Pipeline] BlockIndexer done: items={len(items)} doc={pdf_file_name}")

    # Provider 构建（Stage 3 需要 LLM 调用）
    provider = build_provider_from_custom_config(custom_config)
    provider_info = provider.provider_info()
    try:
        token_counter = provider.token_counter()
    except (AttributeError, NotImplementedError):
        from mineru.custom.enhance_mvp.utils import TokenCounter
        hf_tokenizer = enhance_cfg.get("hf_tokenizer")
        hf_tokenizer = str(hf_tokenizer).strip() if hf_tokenizer not in (None, "") else None
        token_counter = TokenCounter.from_model(model_reference, hf_tokenizer=hf_tokenizer)

    # 获取 API 连接信息（用于 /models API 探测上下文长度）
    base_url = str(enhance_cfg.get("api_base", "")).strip()
    api_key = str(enhance_cfg.get("api_key", "")).strip()

    # 提前获取 context_window 供 Stage 3 使用
    from mineru.custom.enhance_mvp.utils import resolve_context_window
    context_window, ctx_src = resolve_context_window(model_reference, base_url, api_key)
    available_input_tokens = int(context_window * safety_ratio)
    logger.debug(
        f"[Pipeline] context_window={context_window} ({ctx_src}), "
        f"available_input_tokens={available_input_tokens}"
    )

    # Stage 3: LLM Section Planner（替换旧 SectionPlanStage）
    enhance_units, plan_result = run_section_plan_stage(
        items=items,
        provider=provider,
        token_counter=token_counter,
        available_input_tokens=available_input_tokens,
        enhance_cfg=enhance_cfg,
        language=language,
        doc_name=pdf_file_name,
    )
    logger.debug(
        f"[Pipeline] SectionPlan: strategy={plan_result.strategy}, "
        f"{plan_result.original_section_count} titles → {plan_result.enhance_unit_count} units"
    )

    # Stage 4: TokenEstimationStage（flat EnhanceUnit 版本）
    enhance_units, context_window, available_input_tokens = run_token_estimation_stage(
        enhance_units=enhance_units,
        model_reference=model_reference,
        safety_ratio=safety_ratio,
        token_counter=token_counter,
        base_url=base_url,
        api_key=api_key,
    )

    # Stage 5: SectionEnhanceStage（flat，无 bottom-up）
    section_results = run_section_enhance_stage(
        enhance_units=enhance_units,
        provider=provider,
        model_reference=model_reference,
        language=language,
        context_window=context_window,
        available_input_tokens=available_input_tokens,
        concurrency=concurrency,
        token_counter=token_counter,
    )

    # Stage 6: DocEnhanceStage
    overview = run_doc_enhance_stage(
        provider=provider,
        section_results=section_results,
        language=language,
        model_reference=model_reference,
        available_input_tokens=available_input_tokens,
        token_counter=token_counter,
    )

    # Stage 7: ValidateStage
    overview = run_validate_stage(
        doc_overview=overview,
        section_results=section_results,
    )

    # 组装最终 payload
    payload = {
        "doc_name": pdf_file_name,
        "process_mode": process_mode,
        "model_info": {"provider": provider_info.provider, "model": provider_info.model},
        "section_plan": {
            "strategy": plan_result.strategy,
            "doc_type": plan_result.doc_type,
            "fragmentation_rate": round(plan_result.fragmentation_rate, 4),
            "original_section_count": plan_result.original_section_count,
            "enhance_unit_count": len(section_results),
            "fallback": plan_result.fallback,
        },
        "overview": overview,
        "sections": section_results,
    }

    # Stage 8: ComposeStage
    enhanced_md = run_compose_stage(raw_markdown, payload)
    success_count = sum(1 for s in section_results if s.get("status") == "success")
    logger.info(
        f"Enhancement done: doc={pdf_file_name} sections={len(section_results)} "
        f"success={success_count} provider={provider_info.provider} model={provider_info.model}"
    )
    return payload, enhanced_md
