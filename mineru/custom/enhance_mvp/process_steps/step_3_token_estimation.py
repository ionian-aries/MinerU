"""Stage 3: TokenEstimationStage — passthrough。

estimated_tokens 和 char_count 已在 Stage 2 build_enhance_units() 中 inline 计算，
本阶段仅做透传，保留函数签名以避免改动 pipeline 调用链。
"""
from __future__ import annotations

from loguru import logger

from mineru.custom.enhance_mvp.schema import EnhanceUnit
from mineru.custom.enhance_mvp.utils import TokenCounter, resolve_context_window


def run_token_estimation_stage(
    enhance_units: list[EnhanceUnit],
    model_reference: str,
    safety_ratio: float = 0.85,
    token_counter: TokenCounter | None = None,
    base_url: str = "",
    api_key: str = "",
    resolved_context_window: int | None = None,
) -> tuple[list[EnhanceUnit], int, int]:
    """TokenEstimationStage（passthrough 版）：

    estimated_tokens 已在 Stage 2 填写，本阶段仅计算 available_input_tokens 并透传。
    """
    if resolved_context_window is not None and resolved_context_window > 0:
        context_window = resolved_context_window
        src = "pre-resolved"
    else:
        if not isinstance(model_reference, str) or not model_reference.strip():
            raise ValueError("model_reference is required for TokenEstimationStage")
        context_window, src = resolve_context_window(model_reference, base_url, api_key)

    available_input_tokens = int(context_window * float(safety_ratio))
    logger.debug(
        f"[TokenEstimationStage] passthrough: context_window={context_window} ({src}), "
        f"available_input_tokens={available_input_tokens}, units={len(enhance_units)}"
    )
    return enhance_units, context_window, available_input_tokens
