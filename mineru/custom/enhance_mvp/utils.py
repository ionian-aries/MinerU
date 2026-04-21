"""增强 pipeline 工具函数。

包含:
- TokenCounter: 精确/粗估双模式 token 计数器
- TokenBudget: 基于实际测量的精确 token 预算模型
- estimate_tokens: 兜底粗估函数
- split_text_by_token_budget: 按 token 预算切分文本
- resolve_context_window: 五层 fallback 上下文长度获取
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import request as urllib_request

from loguru import logger

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 常见模型名到 HuggingFace ID 的映射
_MODEL_TO_HF: dict[str, str] = {
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2-72b": "Qwen/Qwen2-72B-Instruct",
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "qwen3-235b": "Qwen/Qwen3-235B-A22B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-chat": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "glm-4": "THUDM/glm-4-9b-chat",
    "yi-large": "01-ai/Yi-1.5-34B-Chat",
    "yi-1.5-34b": "01-ai/Yi-1.5-34B-Chat",
}

_LITELLM_JSON_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/"
    "main/model_prices_and_context_window.json"
)
_LITELLM_CACHE_PATH = Path(__file__).resolve().parent / "_model_prices_cache.json"


# ---------------------------------------------------------------------------
# 粗估 token (兜底 fallback)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """粗略估算 token 数量。

    混合文本策略:
    - 中文: 约 1 字 ≈ 1.5 token (偏保守)
    - ASCII 英文: 约 4 字符 ≈ 1 token
    """
    cjk_chars = 0
    ascii_chars = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF      # CJK Unified
            or 0x3400 <= cp <= 0x4DBF   # CJK Extension A
            or 0xF900 <= cp <= 0xFAFF   # CJK Compatibility
            or 0x3000 <= cp <= 0x303F   # CJK Symbols
            or 0xFF00 <= cp <= 0xFFEF   # Fullwidth
            or 0x3040 <= cp <= 0x309F   # Hiragana
            or 0x30A0 <= cp <= 0x30FF   # Katakana
            or 0xAC00 <= cp <= 0xD7AF   # Hangul
        ):
            cjk_chars += 1
        else:
            ascii_chars += 1
    return int(cjk_chars * 1.5) + int(ascii_chars / 4) + 1


# ---------------------------------------------------------------------------
# TokenCounter: 精确/粗估双模式
# ---------------------------------------------------------------------------

class TokenCounter:
    """Token 计数器。优先使用精确 tokenizer，fallback 到粗估。"""

    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer

    @classmethod
    def from_model(cls, model_name: str, *, hf_tokenizer: str | None = None) -> TokenCounter:
        """尝试加载模型对应的 tokenizer。"""
        # 1. 优先使用显式配置（custom-config）
        hf_id = (hf_tokenizer or "").strip()
        # 2. 从模型名推断
        if not hf_id:
            hf_id = _guess_hf_model_id(model_name)
        if hf_id:
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                logger.info(f"[TokenCounter] Loaded precise tokenizer: {hf_id}")
                return cls(tokenizer=tok)
            except Exception as e:
                logger.debug(f"[TokenCounter] AutoTokenizer load failed for {hf_id}: {e}")
        logger.info("[TokenCounter] Using approximate token counter (no precise tokenizer)")
        return cls(tokenizer=None)

    def count(self, text: str) -> int:
        """计算 text 的 token 数。"""
        if not text:
            return 0
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        return estimate_tokens(text)

    def count_messages(self, messages: list[dict]) -> int:
        """计算 chat messages 格式的总 token 数（含 special tokens）。"""
        total = 0
        for msg in messages:
            total += 4  # <|im_start|>role\n ... <|im_end|>\n
            total += self.count(msg.get("content", ""))
        total += 2  # <|im_start|>assistant\n
        return total

    @property
    def is_precise(self) -> bool:
        return self._tokenizer is not None

    @property
    def model_max_length(self) -> int | None:
        """从 tokenizer 获取 model_max_length，不可用时返回 None。"""
        if self._tokenizer is not None:
            val = getattr(self._tokenizer, "model_max_length", None)
            if isinstance(val, int) and 512 < val < 10_000_000:
                return val
        return None


# ---------------------------------------------------------------------------
# 模型名推断
# ---------------------------------------------------------------------------

def _guess_hf_model_id(model_name: str) -> str:
    """从模型名推断 HuggingFace model ID。"""
    if not model_name:
        return ""
    name = model_name.lower().strip()
    if "/" in name:
        parts = name.split("/")
        known_orgs = {"qwen", "deepseek-ai", "thudm", "01-ai", "meta-llama", "mistralai"}
        if parts[0] in known_orgs:
            return model_name  # 保持原始大小写
        name = parts[-1]

    for pattern, hf_id in _MODEL_TO_HF.items():
        if pattern in name:
            return hf_id

    if "/" in model_name and not model_name.startswith("http"):
        return model_name

    return ""


# ---------------------------------------------------------------------------
# 上下文长度获取: 五层 fallback
# ---------------------------------------------------------------------------

def resolve_context_window(model_name: str, base_url: str = "", api_key: str = "") -> tuple[int, str]:
    """五层 fallback 获取模型上下文长度。

    返回 (context_window, source_description)。

    优先级:
    1. transformers AutoTokenizer.model_max_length
    2. litellm model_prices JSON (离线查表)
    3. /models API 运行时探测
    4. 保守兜底 8192
    """
    # Layer 1: transformers tokenizer
    hf_id = _guess_hf_model_id(model_name)
    if hf_id:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            ctx = getattr(tok, "model_max_length", 0)
            if isinstance(ctx, int) and 512 < ctx < 10_000_000:
                return ctx, f"transformers:{hf_id}"
        except Exception as e:
            logger.debug(f"[ContextWindow] transformers fallback failed: {e}")

    # Layer 2: litellm JSON 离线查表
    litellm_ctx = _lookup_litellm_json(model_name)
    if litellm_ctx:
        return litellm_ctx, "litellm_json"

    # Layer 3: /models API 运行时探测
    if base_url and api_key:
        api_ctx = _probe_models_api(base_url, api_key, model_name)
        if api_ctx:
            return api_ctx, "api_probe"

    # Layer 4: 保守兜底
    return 8192, "fallback_default"


def _lookup_litellm_json(model_name: str) -> int | None:
    """查 litellm 社区维护的模型数据库。"""
    data = _load_litellm_data()
    if not data:
        return None

    for candidate in [model_name, model_name.lower()]:
        info = data.get(candidate)
        if info and isinstance(info, dict):
            ctx = info.get("max_input_tokens") or info.get("max_tokens")
            if isinstance(ctx, int) and ctx > 0:
                return ctx

    short = model_name.split("/")[-1].lower()
    for key, val in data.items():
        if not isinstance(val, dict) or val.get("mode") != "chat":
            continue
        if short in key.lower():
            ctx = val.get("max_input_tokens") or val.get("max_tokens")
            if isinstance(ctx, int) and ctx > 0:
                return ctx
    return None


def _load_litellm_data() -> dict:
    """加载 litellm JSON。优先本地缓存，否则远程下载并缓存。"""
    if _LITELLM_CACHE_PATH.exists():
        try:
            age_hours = (time.time() - _LITELLM_CACHE_PATH.stat().st_mtime) / 3600
            if age_hours < 168:  # 7 天有效期
                return json.loads(_LITELLM_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        req = urllib_request.Request(_LITELLM_JSON_URL, method="GET")
        with urllib_request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
        data = json.loads(raw)
        try:
            _LITELLM_CACHE_PATH.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass
        return data
    except Exception as e:
        logger.debug(f"[ContextWindow] litellm JSON download failed: {e}")
        return {}


def _probe_models_api(base_url: str, api_key: str, model_name: str) -> int | None:
    """通过 /models API 探测上下文长度 (vLLM/Ollama 支持)。"""
    try:
        url = f"{base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        req = urllib_request.Request(url=url, headers=headers, method="GET")
        with urllib_request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        models = data.get("data", [])
        for m in models:
            if not isinstance(m, dict):
                continue
            mid = m.get("id", "")
            if model_name.lower() in mid.lower():
                ctx = m.get("max_model_len") or m.get("context_length")
                if isinstance(ctx, int) and ctx > 0:
                    return ctx
    except Exception as e:
        logger.debug(f"[ContextWindow] /models API probe failed: {e}")
    return None


# ---------------------------------------------------------------------------
# TokenBudget: 精确预算模型
# ---------------------------------------------------------------------------

@dataclass
class TokenBudget:
    """基于实际内容测量的精确 token 预算模型。

    所有 overhead 值在初始化时通过 TokenCounter.count() 实际测量，
    而非硬编码常量。
    """
    context_window: int
    safety_ratio: float = 0.90

    system_prompt_tokens: int = 0
    chatml_overhead: int = 12
    schema_hint_tokens: int = 0

    _single_template_tokens: int = 0
    _batch_template_tokens: int = 0
    _batch_section_header_tokens: int = 17
    _doc_template_tokens: int = 0

    def _safe_total(self) -> int:
        return int(self.context_window * self.safety_ratio)

    def single_section_budget(self) -> int:
        fixed = (self.system_prompt_tokens + self.chatml_overhead +
                 self.schema_hint_tokens + self._single_template_tokens)
        output_reserve = 120
        budget = self._safe_total() - fixed - output_reserve
        return max(budget, 500)

    def batch_budget(self, num_sections: int) -> int:
        fixed = (self.system_prompt_tokens + self.chatml_overhead +
                 self.schema_hint_tokens + self._batch_template_tokens)
        per_section_overhead = self._batch_section_header_tokens
        output_reserve = 80 * num_sections + 60
        budget = (self._safe_total() - fixed -
                  per_section_overhead * num_sections - output_reserve)
        return max(budget, 500)

    def doc_enhance_budget(self) -> int:
        fixed = (self.system_prompt_tokens + self.chatml_overhead +
                 self.schema_hint_tokens + self._doc_template_tokens)
        output_reserve = 300
        budget = self._safe_total() - fixed - output_reserve
        return max(budget, 500)

    @classmethod
    def build(
        cls,
        context_window: int,
        token_counter: TokenCounter,
        system_prompt: str,
        schema_hint: str,
        single_template_skeleton: str = "",
        batch_template_skeleton: str = "",
        batch_section_header_sample: str = "",
        doc_template_skeleton: str = "",
        safety_ratio: float | None = None,
    ) -> TokenBudget:
        if safety_ratio is None:
            safety_ratio = 0.90

        return cls(
            context_window=context_window,
            safety_ratio=safety_ratio,
            system_prompt_tokens=token_counter.count(system_prompt),
            schema_hint_tokens=token_counter.count(schema_hint),
            _single_template_tokens=token_counter.count(single_template_skeleton) if single_template_skeleton else 80,
            _batch_template_tokens=token_counter.count(batch_template_skeleton) if batch_template_skeleton else 95,
            _batch_section_header_tokens=token_counter.count(batch_section_header_sample) if batch_section_header_sample else 17,
            _doc_template_tokens=token_counter.count(doc_template_skeleton) if doc_template_skeleton else 71,
        )


# ---------------------------------------------------------------------------
# 文本切分
# ---------------------------------------------------------------------------

def split_text_by_token_budget(text: str, budget: int, token_counter: TokenCounter) -> list[str]:
    """按 token 预算切分文本 (使用 TokenCounter)。"""
    if token_counter.count(text) <= budget:
        return [text]
    rough_chunks = _split_text_impl(text, budget, token_counter.count)
    checked_chunks: list[str] = []
    for chunk in rough_chunks:
        checked_chunks.extend(_enforce_budget(chunk, budget, token_counter.count))
    return checked_chunks


def _enforce_budget(text: str, budget: int, count_fn) -> list[str]:
    if not text:
        return []
    if count_fn(text) <= budget:
        return [text]
    if len(text) <= 1:
        return [text]

    split_at = max(1, len(text) // 2)
    for sep in ("\n", "。", "！", "？", ". ", "! ", "? ", " "):
        pos = text.rfind(sep, max(0, split_at - 20), min(len(text), split_at + 20))
        if pos > 0:
            split_at = pos + len(sep)
            break

    left = text[:split_at]
    right = text[split_at:]
    if not left or not right:
        left = text[: len(text) // 2]
        right = text[len(text) // 2:]

    return _enforce_budget(left, budget, count_fn) + _enforce_budget(right, budget, count_fn)


def _split_text_impl(text: str, budget: int, count_fn) -> list[str]:
    """内部切分实现。"""
    total_chars = len(text)
    total_tokens = count_fn(text)
    if total_tokens == 0:
        return [text]
    chars_per_token = total_chars / total_tokens
    chars_budget = int(budget * chars_per_token)
    if chars_budget <= 0:
        chars_budget = budget

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chars_budget)
        if end < len(text):
            newline_pos = text.rfind("\n", start + chars_budget // 2, end)
            if newline_pos > start:
                end = newline_pos + 1
            else:
                for sep in ("。", "！", "？", ". ", "! ", "? "):
                    sep_pos = text.rfind(sep, start + chars_budget // 2, end)
                    if sep_pos > start:
                        end = sep_pos + len(sep)
                        break
        chunks.append(text[start:end])
        start = end
    return chunks
