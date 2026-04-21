"""LLM 调用客户端，基于 litellm。支持 HTTP 重试、模型 fallback、JSON 格式保障。"""

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any

_MAX_IMAGE_SIDE = 2048  # 对齐 gpt-4 vision tile 边界，超过此值才缩放

# Skip litellm's remote model-cost-map fetch (GitHub CDN; times out in restricted networks).
# We don't use cost-map data — litellm's local backup is sufficient for token counting and
# model inference. Must be set before `import litellm` so it takes effect at module init.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm

from loguru import logger

from mineru.custom.custom_config_loader import CustomConfig
from mineru.custom.enhance_mvp.utils import TokenCounter, resolve_context_window


class LLMCallError(RuntimeError):
    """litellm 所有 retry + fallback 耗尽后仍失败。"""


class JSONParseError(ValueError):
    """两次 JSON 解析尝试均失败（含代码块清洗）。"""


@dataclass
class ProviderInfo:
    provider: str
    model: str


@dataclass
class FallbackModel:
    """备用模型。api_base / api_key 为空时复用主模型 endpoint。"""

    name: str
    api_base: str = ""
    api_key: str = ""


class LLMClient:
    """基于 litellm 的 LLM 调用客户端。HTTP 重试/fallback 和 JSON 格式保障。"""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        *,
        timeout_s: int = 300,
        http_retries: int = 2,
        hf_tokenizer: str | None = None,
        context_window_override: int | None = None,
        max_output_tokens: int | None = None,
        fallback_models: list[FallbackModel] | None = None,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = max(1, int(timeout_s))
        self._http_retries = max(0, int(http_retries))
        self._max_output_tokens = max_output_tokens
        self._fallback_models: list[FallbackModel] = fallback_models or []

        self._token_counter = TokenCounter.from_model(model, hf_tokenizer=hf_tokenizer)
        self._ctx_window, ctx_src = self._resolve_ctx(context_window_override)

        logger.info(
            f"[LLMClient] {model} ctx={self._ctx_window}({ctx_src}) "
            f"tokenizer={'precise' if self._token_counter.is_precise else 'approx'}"
            + (
                f" fallbacks={[m.name for m in self._fallback_models]}"
                if self._fallback_models
                else ""
            )
        )

    def _resolve_ctx(self, override: int | None) -> tuple[int, str]:
        """优先级：config override → tokenizer.model_max_length → 五层 fallback。"""
        if isinstance(override, int) and override > 0:
            return override, "custom_config"
        if self._token_counter.model_max_length:
            return self._token_counter.model_max_length, "tokenizer"
        return resolve_context_window(self._model, self._base_url, self._api_key)

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        """提取 JSON，处理 ```json...``` 包装和前后说明文字。失败抛 JSONDecodeError。"""
        s = content.strip()
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s)
        if m:
            return json.loads(m.group(1))
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end > start:
            return json.loads(s[start : end + 1])
        return json.loads(s)

    def _do_completion(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        *,
        json_mode: bool = True,
    ) -> str:
        """唯一的 litellm HTTP 调用点。失败抛 LLMCallError。"""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "api_base": self._base_url,
            "api_key": self._api_key,
            "messages": messages,
            "temperature": 0.2,
            "timeout": self._timeout_s,
            "num_retries": self._http_retries,
            "drop_params": True,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        effective_max = max_tokens or self._max_output_tokens
        if effective_max:
            kwargs["max_tokens"] = effective_max
        if self._fallback_models:
            kwargs["fallbacks"] = [
                {
                    "model": m.name,
                    "api_base": m.api_base or self._base_url,
                    "api_key": m.api_key or self._api_key,
                }
                for m in self._fallback_models
            ]
        try:
            resp = litellm.completion(**kwargs)
            choice = resp.choices[0]
            content = choice.message.content or ""
            if getattr(choice, "finish_reason", None) == "length":
                if json_mode:
                    raise LLMCallError("output truncated by max_tokens limit")
                # json_mode=False（如图片描述）：截断文字仍可用，由调用方质量检查决定
            return content
        except LLMCallError:
            raise
        except Exception as exc:
            raise LLMCallError(f"LLM call failed: {exc}") from exc

    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(provider="openai_compatible", model=self._model)

    def token_counter(self) -> TokenCounter:
        return self._token_counter

    @staticmethod
    def _prepare_image_b64(image_path: str) -> tuple[str, str]:
        """读取本地图片，必要时缩放，返回 (base64字符串, mime_type)。

        策略：
        - 最长边 ≤ 2048px：直接读原始字节 base64，零重编码，格式/质量完全保真。
        - 最长边 > 2048px：等比缩放到 longest=2048，LANCZOS 重采样，保留原始格式。
        """
        from PIL import Image  # 仅在此处 lazy import，避免顶层强依赖

        _FMT_TO_MIME: dict[str, str] = {
            "JPEG": "image/jpeg",
            "JPG":  "image/jpeg",
            "PNG":  "image/png",
            "WEBP": "image/webp",
            "GIF":  "image/gif",
        }

        with Image.open(image_path) as img:
            fmt = img.format or "PNG"          # 在任何操作前捕获（resize 后 format 属性丢失）
            mime = _FMT_TO_MIME.get(fmt.upper(), "image/png")
            w, h = img.size

            if max(w, h) <= _MAX_IMAGE_SIDE:
                # 无需缩放：绕过 PIL 重编码，直接读原始字节
                with open(image_path, "rb") as f:
                    raw = f.read()
                return base64.b64encode(raw).decode(), mime

            # 需要缩放：等比缩放，保留原始格式写入内存 buffer
            scale = _MAX_IMAGE_SIDE / max(w, h)
            img_resized = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img_resized.save(buf, format=fmt)
            return base64.b64encode(buf.getvalue()).decode(), mime

    def generate_with_image(
        self,
        *,
        prompt: str,
        image_path: str,
        system_prompt: str,
        max_tokens: int,
    ) -> str:
        """多模态调用：发送图片+文本 prompt，返回纯文本描述。

        - 图片以 base64 data URL 内嵌，detail=high 保证全精度 tile 分析。
        - 不使用 json_mode，返回自由文本（非 JSON）。
        """
        b64, mime = self._prepare_image_b64(image_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return self._do_completion(messages, max_tokens=max_tokens, json_mode=False)

    def generate_json(
        self,
        prompt: str,
        system_prompt: str,
        schema_hint: str = "",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """调用 LLM 返回合法 dict。JSON 解析失败时强化指令重试一次，仍失败抛 JSONParseError。"""
        if schema_hint:
            prompt = f"{prompt}\n\n你的输出必须严格匹配以下JSON结构：\n{schema_hint}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        content = self._do_completion(messages, max_tokens=max_tokens)
        try:
            return self._extract_json(content)
        except (json.JSONDecodeError, ValueError):
            logger.warning("[LLMClient] JSON parse failed, retrying")

        strict_sys = (
            system_prompt
            + "\n【严格要求】只允许输出合法的JSON对象，禁止任何代码块标记（如```）、注释或额外文字。"
        )
        content2 = self._do_completion(
            [
                {"role": "system", "content": strict_sys},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        try:
            return self._extract_json(content2)
        except (json.JSONDecodeError, ValueError) as exc:
            raise JSONParseError(
                f"JSON parse failed after format retry. snippet: {content2[:200]!r}"
            ) from exc


def build_provider_from_custom_config(custom_config: CustomConfig) -> LLMClient:
    enhance = custom_config.enhance or {}
    provider_type = str(enhance.get("provider", "openai_compatible")).strip().lower()
    if provider_type not in {"openai", "openai_compatible"}:
        raise ValueError(
            f"Unsupported enhance.provider='{provider_type}'. Supported: openai_compatible."
        )

    model = str(enhance.get("model", "")).strip()
    base_url = str(enhance.get("api_base", "")).strip()
    api_key = str(enhance.get("api_key", "")).strip()
    missing = [
        f"enhance.{k}"
        for k, v in [("model", model), ("api_base", base_url), ("api_key", api_key)]
        if not v
    ]
    if missing:
        raise ValueError("Enhancement provider config missing: " + ", ".join(missing))

    timeout_s = int(float(enhance.get("timeout_s", 300) or 300))
    http_retries = int(float(enhance.get("http_retries", 2) or 2))

    hf_tokenizer = enhance.get("hf_tokenizer")
    hf_tokenizer = str(hf_tokenizer).strip() if hf_tokenizer not in (None, "") else None

    ctx_win = enhance.get("context_window")
    context_window_override = int(ctx_win) if isinstance(ctx_win, int) else None

    max_out = enhance.get("max_output_tokens")
    max_output_tokens = int(max_out) if max_out else None

    fallback_models: list[FallbackModel] = []
    for item in enhance.get("fallback_models") or []:
        if isinstance(item, str) and item.strip():
            fallback_models.append(FallbackModel(name=item.strip()))
        elif isinstance(item, dict) and (name := str(item.get("name", "")).strip()):
            fallback_models.append(
                FallbackModel(
                    name=name,
                    api_base=str(item.get("api_base", "") or "").strip(),
                    api_key=str(item.get("api_key", "") or "").strip(),
                )
            )

    return LLMClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_s=max(1, timeout_s),
        http_retries=max(0, http_retries),
        hf_tokenizer=hf_tokenizer,
        context_window_override=context_window_override,
        max_output_tokens=max_output_tokens,
        fallback_models=fallback_models,
    )
