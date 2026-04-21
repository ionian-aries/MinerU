import json
import time
from dataclasses import dataclass
from typing import Any
from urllib import error as urlerror
from urllib import request

from loguru import logger

from mineru.custom.enhance_mvp.utils import TokenCounter, resolve_context_window
from mineru.custom.custom_config_loader import CustomConfig


@dataclass
class ProviderInfo:
    provider: str
    model: str


class BaseEnhanceProvider:
    def provider_info(self) -> ProviderInfo:
        raise NotImplementedError

    def generate_json(self, prompt: str, system_prompt: str, schema_hint: str = "") -> dict[str, Any]:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return len(text)

    def model_name(self) -> str:
        return self.provider_info().model

    def provider_name(self) -> str:
        return self.provider_info().provider

    def context_window(self) -> int:
        """模型的上下文窗口大小 (tokens)。"""
        return 8192

    def max_output_tokens(self) -> int:
        """模型的最大输出 token 数。0 表示不确定。"""
        return 0

    def token_counter(self) -> TokenCounter:
        """返回绑定到该 provider 的 token 计数器。"""
        return TokenCounter()


class OpenAICompatibleProvider(BaseEnhanceProvider):
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
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._http_retries = max(0, int(http_retries))
        self._context_window_override = context_window_override

        # 初始化时一次性解析上下文长度和 tokenizer
        self._token_counter = TokenCounter.from_model(self._model, hf_tokenizer=hf_tokenizer)
        self._context_window, self._ctx_source = self._resolve_ctx()

        logger.info(
            f"[Provider] model={self._model}, "
            f"context_window={self._context_window} (from {self._ctx_source}), "
            f"tokenizer={'precise' if self._token_counter.is_precise else 'approx'}"
        )

    def _resolve_ctx(self) -> tuple[int, str]:
        """解析上下文长度。优先用已加载的 tokenizer，否则走统一五层 fallback。"""
        if isinstance(self._context_window_override, int) and self._context_window_override > 0:
            return self._context_window_override, "custom_config:context_window"
        # 优先使用已加载 tokenizer 的 model_max_length（避免重复加载）
        if self._token_counter.model_max_length:
            return self._token_counter.model_max_length, f"tokenizer:{self._model}"

        # 统一五层 fallback（env → transformers → litellm JSON → /models API → 8192）
        return resolve_context_window(self._model, self._base_url, self._api_key)

    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(provider="openai_compatible", model=self._model)

    def context_window(self) -> int:
        return self._context_window

    def max_output_tokens(self) -> int:
        return 0

    def token_counter(self) -> TokenCounter:
        return self._token_counter

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        req = request.Request(url=url, data=body, headers=headers, method="POST")

        max_retries = self._http_retries
        retry_codes = {429, 500, 502, 503, 504}
        for attempt in range(max_retries + 1):
            try:
                with request.urlopen(req, timeout=self._timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urlerror.HTTPError as exc:
                status = getattr(exc, "code", None)
                body_bytes = exc.read()
                err_body = body_bytes.decode("utf-8", errors="replace")[:1000]
                should_retry = status in retry_codes and attempt < max_retries
                if should_retry:
                    wait = 0.8 * (2 ** attempt)
                    if status == 429:
                        retry_after = exc.headers.get("Retry-After") if hasattr(exc, "headers") else None
                        if retry_after:
                            try:
                                wait = max(wait, float(retry_after))
                            except ValueError:
                                pass
                    logger.warning(f"[Provider] HTTP {status}, retry in {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Enhancement HTTP error status={status}, url={url}, body={err_body}"
                ) from exc
            except (urlerror.URLError, TimeoutError) as exc:
                if attempt < max_retries:
                    time.sleep(0.8 * (2 ** attempt))
                    continue
                raise RuntimeError(f"Enhancement network error url={url}: {exc}") from exc

    def _chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            from litellm import completion  # type: ignore

            response = completion(
                model=self._model,
                api_base=self._base_url,
                api_key=self._api_key,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=self._timeout_s,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except ModuleNotFoundError:
            payload = {
                "model": self._model,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            response_json = self._post_chat_completion(payload)
            content = response_json["choices"][0]["message"]["content"]
            return json.loads(content)

    def generate_json(self, prompt: str, system_prompt: str, schema_hint: str = "") -> dict[str, Any]:
        if schema_hint:
            prompt = f"{prompt}\n\n你的输出必须严格匹配以下JSON结构：\n{schema_hint}"
        return self._chat_json(system_prompt, prompt)


def build_provider_from_custom_config(custom_config: CustomConfig) -> BaseEnhanceProvider:
    enhance = custom_config.enhance or {}
    provider = str(enhance.get("provider", "openai_compatible")).strip().lower()
    if provider not in {"openai", "openai_compatible"}:
        raise ValueError(
            f"Unsupported enhance.provider='{provider}'. Supported: openai_compatible."
        )

    model = str(enhance.get("model", "")).strip()
    base_url = str(enhance.get("api_base", "")).strip()
    api_key = str(enhance.get("api_key", "")).strip()
    timeout_s = int(float(enhance.get("timeout_s", 300) or 300))
    http_retries = int(float(enhance.get("http_retries", 2) or 2))
    hf_tokenizer = enhance.get("hf_tokenizer")
    hf_tokenizer = str(hf_tokenizer).strip() if hf_tokenizer not in (None, "") else None
    context_window = enhance.get("context_window")
    context_window_override = int(context_window) if isinstance(context_window, int) else None

    missing = []
    if not model:
        missing.append("enhance.model")
    if not base_url:
        missing.append("enhance.api_base")
    if not api_key:
        missing.append("enhance.api_key")
    if missing:
        raise ValueError(
            "Enhancement provider config missing in custom config: " + ", ".join(missing)
        )

    return OpenAICompatibleProvider(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_s=max(1, timeout_s),
        http_retries=max(0, http_retries),
        hf_tokenizer=hf_tokenizer,
        context_window_override=context_window_override,
    )
