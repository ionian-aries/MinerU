"""Stage 3: TokenEstimationStage — 计算每个 EnhanceUnit 的 token 数，获取上下文长度。

重构版：输入从 SectionNode tree 改为 flat list[EnhanceUnit]。
其余逻辑（TokenCounter / resolve_context_window）不变。
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
) -> tuple[list[EnhanceUnit], int, int]:
    """TokenEstimationStage（重构版）：

    - 获取上下文长度 context_window（统一五层 fallback）
    - 逐 unit 计算 estimated_tokens（基于 body_text）
    - 计算 available_input_tokens（安全余量后的上下文上限）

    参数:
        enhance_units     — LLM Section Planner 产出的 EnhanceUnit 列表
        model_reference   — 用于查询上下文长度和 tokenizer 的模型名
        safety_ratio      — 安全余量系数（默认 0.85）
        token_counter     — 可选，复用已创建的 TokenCounter 实例
        base_url          — API 端点（用于 /models API 探测上下文长度）
        api_key           — API key

    返回:
        (enhance_units_with_tokens, context_window, available_input_tokens)
    """
    if not isinstance(model_reference, str) or not model_reference.strip():
        raise ValueError("model_reference is required for TokenEstimationStage")

    if token_counter is None:
        token_counter = TokenCounter.from_model(model_reference)

    context_window, src = resolve_context_window(model_reference, base_url, api_key)
    available_input_tokens = int(context_window * float(safety_ratio))

    logger.debug(
        f"[TokenEstimationStage] context_window={context_window} ({src}), "
        f"available_input_tokens={available_input_tokens}, safety_ratio={safety_ratio}, "
        f"tokenizer={'precise' if token_counter.is_precise else 'approx'}"
    )

    for unit in enhance_units:
        unit.estimated_tokens = token_counter.count(unit.body_text)

    total_tokens = sum(u.estimated_tokens for u in enhance_units)
    logger.debug(
        f"[TokenEstimationStage] units={len(enhance_units)} total_body_tokens={total_tokens}"
    )

    return enhance_units, context_window, available_input_tokens
