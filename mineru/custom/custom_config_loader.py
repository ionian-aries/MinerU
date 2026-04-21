from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CustomConfig:
    """二开唯一配置源（服务级/命令级固定配置）。

    约束：
    - 仅从用户显式指定的文件加载（YAML 主格式，兼容 JSON）。
    - 不做任何环境变量覆盖或 ${ENV_VAR} 展开。
    - 通过 validate_custom_config() 做 fail-fast 校验。
    """

    enhance: dict[str, Any]
    storage: dict[str, Any]
    discard: dict[str, Any]

    @classmethod
    def disabled(cls) -> "CustomConfig":
        return cls(enhance={"enable": False}, storage={}, discard={})


def load_custom_config(path: str | Path) -> CustomConfig:
    p = Path(path)
    raw_text = p.read_text(encoding="utf-8")

    data: Any
    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyYAML is required to load .yaml custom config files. "
                "Please install PyYAML or use a .json config."
            ) from exc
        data = yaml.safe_load(raw_text)
    elif suffix == ".json":
        data = json.loads(raw_text)
    else:
        # YAML loader通常可兼容 JSON；这里为了可预期性，按后缀做严格分流。
        raise ValueError(
            f"Unsupported custom config format: {p.name}. "
            "Use .yaml/.yml (recommended) or .json."
        )

    if not isinstance(data, dict):
        raise ValueError("custom config root must be a mapping/object")

    enhance = data.get("enhance") or {}
    storage = data.get("storage") or {}
    discard = data.get("discard") or {}
    if not isinstance(enhance, dict) or not isinstance(storage, dict) or not isinstance(discard, dict):
        raise ValueError("custom config sections enhance/storage/discard must be objects")

    cfg = CustomConfig(enhance=enhance, storage=storage, discard=discard)
    validate_custom_config(cfg)
    return cfg


def validate_custom_config(cfg: CustomConfig) -> None:
    # --- enhance ---
    enhance = cfg.enhance or {}
    enable = bool(enhance.get("enable", False))
    if enable:
        missing: list[str] = []
        for k in ("model", "api_base", "api_key"):
            v = enhance.get(k)
            if not isinstance(v, str) or not v.strip():
                missing.append(f"enhance.{k}")
        if missing:
            raise ValueError(
                "CustomConfig validation failed: enhance.enable=true but missing "
                + ", ".join(missing)
            )

    # --- enhance.section_plan ---
    section_plan = enhance.get("section_plan")
    if section_plan is not None:
        if isinstance(section_plan, str):
            # 简写形式: section_plan: "auto" / "off" / "always"
            if section_plan.strip().lower() not in {"auto", "off", "always"}:
                raise ValueError(
                    "CustomConfig validation failed: enhance.section_plan must be "
                    "'auto', 'off', 'always', or a config object"
                )
        elif isinstance(section_plan, dict):
            sp_enabled = str(section_plan.get("enabled", "auto")).strip().lower()
            if sp_enabled not in {"auto", "off", "always"}:
                raise ValueError(
                    "CustomConfig validation failed: enhance.section_plan.enabled must be "
                    "'auto', 'off', or 'always'"
                )
            sp_threshold = section_plan.get("threshold")
            if sp_threshold is not None:
                try:
                    t = float(sp_threshold)
                    if not (0.0 <= t <= 1.0):
                        raise ValueError
                except (TypeError, ValueError):
                    raise ValueError(
                        "CustomConfig validation failed: enhance.section_plan.threshold "
                        "must be a float between 0.0 and 1.0"
                    )
            sp_min = section_plan.get("min_chars")
            if sp_min is not None:
                try:
                    m = int(sp_min)
                    if m < 1:
                        raise ValueError
                except (TypeError, ValueError):
                    raise ValueError(
                        "CustomConfig validation failed: enhance.section_plan.min_chars "
                        "must be a positive integer"
                    )
            sp_max = section_plan.get("max_chars")
            if sp_max is not None:
                try:
                    mx = int(sp_max)
                    if mx < 1:
                        raise ValueError
                except (TypeError, ValueError):
                    raise ValueError(
                        "CustomConfig validation failed: enhance.section_plan.max_chars "
                        "must be a positive integer"
                    )
        else:
            raise ValueError(
                "CustomConfig validation failed: enhance.section_plan must be "
                "a string ('auto'/'off'/'always') or a config object"
            )

    # --- storage ---
    storage = cfg.storage or {}
    for scope in ("image", "doc"):
        node = storage.get(scope)
        if node is None:
            continue
        if not isinstance(node, dict):
            raise ValueError(f"CustomConfig validation failed: storage.{scope} must be an object")
        backend = str(node.get("backend", "local")).strip().lower()
        if backend not in {"local", "s3"}:
            raise ValueError(
                f"CustomConfig validation failed: storage.{scope}.backend must be 'local' or 's3'"
            )
        if backend == "s3":
            required = ("s3_bucket", "s3_ak", "s3_sk", "s3_endpoint_url")
            missing = [f"storage.{scope}.{k}" for k in required if not str(node.get(k, "")).strip()]
            if missing:
                raise ValueError(
                    "CustomConfig validation failed: backend=s3 but missing "
                    + ", ".join(missing)
                )

    # --- discard ---
    discard = cfg.discard or {}
    types = discard.get("types")
    if types is None:
        return
    if not isinstance(types, list) or not all(isinstance(x, str) and x.strip() for x in types):
        raise ValueError("CustomConfig validation failed: discard.types must be a list of strings")

