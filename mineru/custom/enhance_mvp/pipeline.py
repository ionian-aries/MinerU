"""Enhancement pipeline 主编排入口（重构版）。

7 阶段（notes/20）:
  Stage 1: BlockIndexer（替换 SectionBuildStage）
  Stage 2: LLM Section Planner（替换旧 SectionPlanStage）
  Stage 3: TokenEstimationStage（flat EnhanceUnit）
  Stage 4: SectionEnhanceStage（废弃 bottom-up）
  Stage 5: DocEnhanceStage（含 answerable_questions）
  Stage 6: ValidateStage
  Stage 7: ComposeStage
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from mineru.custom.custom_config_loader import CustomConfig
from mineru.custom.enhance_mvp.provider import build_provider_from_custom_config
from mineru.custom.enhance_mvp.process_steps import (
    build_block_items,
    run_compose_stage,
    run_doc_enhance_stage,
    run_image_enhance_stage,
    run_section_enhance_stage,
    run_section_plan_stage,
    run_token_estimation_stage,
    run_validate_stage,
)


def run_enhancement_pipeline(
    *,
    custom_config: CustomConfig,
    pdf_file_name: str,
    process_mode: str,
    middle_json: dict[str, Any],
    content_list_v2: Any,
    raw_markdown: str,
    image_dir: str = "",
) -> tuple[dict[str, Any], str]:
    enhance_cfg = custom_config.enhance or {}
    language = str(enhance_cfg.get("language", "zh") or "zh")
    model = str(enhance_cfg.get("model", "")).strip()
    model_reference = str(enhance_cfg.get("model_reference", "")).strip() or model
    if not model_reference:
        raise ValueError("Enhancement model_reference is empty in custom config.")
    safety_ratio = float(enhance_cfg.get("safety_ratio", 0.85) or 0.85)
    concurrency = int(enhance_cfg.get("concurrency", 4) or 4)

    # Step 1: Block Indexer（替换 SectionBuildStage）
    items = build_block_items(middle_json)
    logger.debug(f"[Pipeline] BlockIndexer done: items={len(items)} doc={pdf_file_name}")

    # Provider 构建（Stage 2 需要 LLM 调用）
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

    # Step 1.5: ImageEnhanceStage（可选，enabled 开关控制）
    image_enhance_cfg = enhance_cfg.get("image_enhance") or {}
    image_enhance_records: dict[str, dict] = {}
    if image_enhance_cfg.get("enabled", False):
        items, image_enhance_records = run_image_enhance_stage(
            items=items,
            provider=provider,
            image_enhance_cfg=image_enhance_cfg,
            image_dir=image_dir,
        )
        # 补充 model 信息（Stage 1.5 内部无法直接访问 model 名称）
        for rec in image_enhance_records.values():
            rec["params"]["model"] = model
        logger.debug(f"[Pipeline] ImageEnhance done: doc={pdf_file_name}")

    # Step 2: LLM Section Planner（替换旧 SectionPlanStage）
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

    # Step 3: TokenEstimationStage（复用 Stage 2 前已解析的 available_input_tokens）
    enhance_units, context_window, available_input_tokens = run_token_estimation_stage(
        enhance_units=enhance_units,
        model_reference=model_reference,
        safety_ratio=safety_ratio,
        token_counter=token_counter,
        base_url=base_url,
        api_key=api_key,
        resolved_context_window=context_window,
    )

    # Step 4: SectionEnhanceStage（flat，无 bottom-up）
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

    # 后处理：根据 refs 内置 bbox 字符串写入 pos_range
    # - refs 非 image_source：形如 [page_no, block_no, type, "x0,y0,x1,y1"]
    # - image_source 虚拟 ref：保持 3 元素（无真实 bbox）
    # - pos_range.start = [page_no, y0]（首个真实 ref 的顶边 y）
    # - pos_range.end   = [page_no, y1]（末个真实 ref 的底边 y）
    for _sec in section_results:
        _refs = _sec.get("refs") or []
        real_refs: list[list] = []
        for _ref in _refs:
            if not isinstance(_ref, list) or len(_ref) < 4:
                continue
            if _ref[2] == "image_source":
                continue
            if not isinstance(_ref[3], str) or "," not in _ref[3]:
                continue
            real_refs.append(_ref)

        if real_refs:
            def _parse_bbox(bbox_str: str) -> tuple[int, int, int, int] | None:
                try:
                    x0, y0, x1, y1 = (int(float(p)) for p in bbox_str.split(",")[:4])
                    return x0, y0, x1, y1
                except Exception:
                    return None

            first_bbox = _parse_bbox(real_refs[0][3])
            last_bbox = _parse_bbox(real_refs[-1][3])
            _sec["pos_range"] = {
                "start": [real_refs[0][0], int(first_bbox[1]) if first_bbox else 0],
                "end":   [real_refs[-1][0], int(last_bbox[3]) if last_bbox else 0],
            }
        else:
            _sec["pos_range"] = {"start": [-1, 0], "end": [-1, 0]}

    # Step 5: DocEnhanceStage
    overview = run_doc_enhance_stage(
        provider=provider,
        section_results=section_results,
        language=language,
        model_reference=model_reference,
        available_input_tokens=available_input_tokens,
        token_counter=token_counter,
    )

    # Step 6: ValidateStage
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
            "fallback_reason": plan_result.fallback_reason,
        },
        "overview": overview,
        "sections": section_results,
        "image_enhancements": image_enhance_records,
    }

    # Step 7: ComposeStage
    split_marker = str(enhance_cfg.get("split_marker") or "")
    enhanced_md = run_compose_stage(raw_markdown, payload, split_marker=split_marker)
    success_count = sum(1 for s in section_results if s.get("status") == "success")
    logger.info(
        f"Enhancement done: doc={pdf_file_name} sections={len(section_results)} "
        f"success={success_count} provider={provider_info.provider} model={provider_info.model}"
    )
    return payload, enhanced_md
