from __future__ import annotations

from typing import Optional

from mineru.custom.custom_config_loader import CustomConfig
from mineru.custom.discard_policy.discard_policy import DiscardPolicy
from mineru.custom.enhance_mvp import run_enhancement_pipeline
from mineru.custom.enhance_mvp.provider import build_provider_from_custom_config
from mineru.custom.storage.storage import (
    build_output_writers_from_custom_config,
    resolve_image_ref_prefix_from_custom_config,
)


def resolve_discard_policy(discard_types=None) -> DiscardPolicy:
    """Resolve discard policy from explicit args only (no env fallback)."""
    return DiscardPolicy.from_discard_types(discard_types)


def resolve_output_writers(
    local_image_dir,
    local_md_dir,
    *,
    custom_config: Optional[CustomConfig] = None,
):
    """Output writers resolution.

    Contract:
    - Without custom_config: behave like original MinerU (local writers only).
    - With custom_config: use custom_config.storage as the only source of truth.
    """
    if custom_config is None:
        from mineru.data.data_reader_writer import FileBasedDataWriter

        return FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
    return build_output_writers_from_custom_config(
        local_image_dir, local_md_dir, custom_config=custom_config
    )


def resolve_image_output_ref_prefix(
    local_image_dir: str,
    *,
    custom_config: Optional[CustomConfig] = None,
) -> str:
    """Markdown image reference prefix resolution."""
    if custom_config is None:
        import os

        return str(os.path.basename(local_image_dir))
    return resolve_image_ref_prefix_from_custom_config(
        local_image_dir, custom_config=custom_config
    )


def resolve_enhancement_enabled(*, custom_config: Optional[CustomConfig] = None) -> bool:
    """Enhancement enable switch (custom-config only)."""
    if custom_config is None:
        return False
    return bool(custom_config.enhance.get("enable", False))


def resolve_enhancement_provider(*, custom_config: Optional[CustomConfig] = None):
    """Resolve and validate enhancement provider.

    NOTE: This call will raise if provider config is incomplete.
    """
    if custom_config is None:
        raise ValueError("Enhancement is disabled (missing custom_config).")
    return build_provider_from_custom_config(custom_config)


def resolve_enhancement_runner(*, custom_config: Optional[CustomConfig] = None):
    """Return enhancement pipeline runner bound to custom_config."""
    if custom_config is None:
        raise ValueError("Enhancement is disabled (missing custom_config).")
    return lambda **kwargs: run_enhancement_pipeline(custom_config=custom_config, **kwargs)
