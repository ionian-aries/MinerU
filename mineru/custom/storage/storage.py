from typing import Any

from mineru.data.data_reader_writer import DataWriter, FileBasedDataWriter, S3DataWriter
from mineru.custom.custom_config_loader import CustomConfig


def _to_posix_path(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def build_output_writers(
    local_image_dir: str,
    local_md_dir: str,
    *,
    storage_options: dict[str, Any] | None = None,
) -> tuple[DataWriter, DataWriter]:
    """Legacy signature kept for internal callers, but with NO env fallbacks.

    This function reads ONLY from storage_options. If storage_options is None,
    it behaves like original MinerU local writers.
    """
    if not isinstance(storage_options, dict):
        return FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

    def _node(scope: str) -> dict[str, Any]:
        node = storage_options.get(scope)
        return node if isinstance(node, dict) else {"backend": "local"}

    def _backend(node: dict[str, Any]) -> str:
        return str(node.get("backend", "local")).strip().lower()

    def _writer(scope: str, node: dict[str, Any], local_dir: str) -> DataWriter:
        backend = _backend(node)
        if backend == "local":
            return FileBasedDataWriter(local_dir)
        if backend == "s3":
            required = ("s3_bucket", "s3_ak", "s3_sk", "s3_endpoint_url")
            missing = [k for k in required if str(node.get(k, "")).strip() == ""]
            if missing:
                raise ValueError(
                    f"storage {scope} backend=s3 missing fields: {', '.join(missing)}"
                )
            prefix = str(node.get("s3_prefix", "") or _to_posix_path(local_dir))
            addressing_style = str(node.get("s3_addressing_style", "auto") or "auto")
            return S3DataWriter(
                default_prefix_without_bucket=_to_posix_path(prefix),
                bucket=str(node["s3_bucket"]).strip(),
                ak=str(node["s3_ak"]).strip(),
                sk=str(node["s3_sk"]).strip(),
                endpoint_url=str(node["s3_endpoint_url"]).strip(),
                addressing_style=addressing_style,
            )
        raise ValueError(
            f"Unsupported storage backend for {scope}: {backend}. Supported: local, s3"
        )

    return (
        _writer("image", _node("image"), local_image_dir),
        _writer("doc", _node("doc"), local_md_dir),
    )


def resolve_image_ref_prefix(
    local_image_dir: str,
    *,
    storage_options: dict[str, Any] | None = None,
) -> str:
    """Legacy signature kept for internal callers, but with NO env fallbacks."""
    if isinstance(storage_options, dict):
        image_node = storage_options.get("image")
        if isinstance(image_node, dict):
            ref_prefix = image_node.get("ref_prefix")
            if ref_prefix not in (None, ""):
                return str(ref_prefix).strip().rstrip("/")
    import os

    return str(os.path.basename(local_image_dir))


def build_output_writers_from_custom_config(
    local_image_dir: str,
    local_md_dir: str,
    *,
    custom_config: CustomConfig,
) -> tuple[DataWriter, DataWriter]:
    storage_options = custom_config.storage or {}
    image_node = storage_options.get("image") if isinstance(storage_options, dict) else None
    doc_node = storage_options.get("doc") if isinstance(storage_options, dict) else None
    if not isinstance(image_node, dict):
        image_node = {"backend": "local"}
    if not isinstance(doc_node, dict):
        doc_node = {"backend": "local"}

    def _backend(node: dict[str, Any]) -> str:
        return str(node.get("backend", "local")).strip().lower()

    def _writer(scope: str, node: dict[str, Any], local_dir: str) -> DataWriter:
        backend = _backend(node)
        if backend == "local":
            return FileBasedDataWriter(local_dir)
        if backend == "s3":
            required = ("s3_bucket", "s3_ak", "s3_sk", "s3_endpoint_url")
            missing = [k for k in required if str(node.get(k, "")).strip() == ""]
            if missing:
                raise ValueError(
                    f"custom storage {scope} backend=s3 missing fields: {', '.join(missing)}"
                )
            prefix = str(node.get("s3_prefix", "") or _to_posix_path(local_dir))
            addressing_style = str(node.get("s3_addressing_style", "auto") or "auto")
            return S3DataWriter(
                default_prefix_without_bucket=_to_posix_path(prefix),
                bucket=str(node["s3_bucket"]).strip(),
                ak=str(node["s3_ak"]).strip(),
                sk=str(node["s3_sk"]).strip(),
                endpoint_url=str(node["s3_endpoint_url"]).strip(),
                addressing_style=addressing_style,
            )
        raise ValueError(
            f"Unsupported custom storage backend for {scope}: {backend}. Supported: local, s3"
        )

    return (
        _writer("image", image_node, local_image_dir),
        _writer("doc", doc_node, local_md_dir),
    )


def resolve_image_ref_prefix_from_custom_config(
    local_image_dir: str,
    *,
    custom_config: CustomConfig,
) -> str:
    storage_options = custom_config.storage or {}
    image_node = storage_options.get("image") if isinstance(storage_options, dict) else None
    if isinstance(image_node, dict):
        ref_prefix = image_node.get("ref_prefix")
        if ref_prefix not in (None, ""):
            return str(ref_prefix).strip().rstrip("/")
    return str(os.path.basename(local_image_dir))
