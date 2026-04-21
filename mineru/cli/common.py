# Copyright (c) Opendatalab. All rights reserved.
import importlib
import importlib.util
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from loguru import logger

from mineru.custom.registry import (
    resolve_enhancement_enabled,
    resolve_image_output_ref_prefix,
    resolve_output_writers,
)
from mineru.custom.enhance_mvp import run_enhancement_pipeline
from mineru.custom.custom_config_loader import CustomConfig
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.enum_class import MakeMode
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_bytes
from mineru.utils.pdf_image_tools import images_bytes_to_pdf_bytes
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.backend.office.office_middle_json_mkcontent import union_make as office_union_make
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_analyze import aio_doc_analyze as aio_vlm_doc_analyze
from mineru.backend.office.docx_analyze import office_docx_analyze
from mineru.custom.registry import resolve_discard_policy
from mineru.utils.pdfium_guard import rewrite_pdf_bytes_with_pdfium

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
if os.getenv("MINERU_LMDEPLOY_DEVICE", "") == "maca":
    import torch
    torch.backends.cudnn.enabled = False


pdf_suffixes = ["pdf"]
image_suffixes = ["png", "jpeg", "jp2", "webp", "gif", "bmp", "jpg", "tiff"]
docx_suffixes = ["docx"]
pptx_suffixes = ["pptx"]
xlsx_suffixes = ["xlsx"]
office_suffixes = docx_suffixes + pptx_suffixes + xlsx_suffixes

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Maximum UTF-8 byte length allowed for task stems used in filenames.
# 200 bytes is chosen to stay well below common filesystem limits (e.g. 255 bytes)
# and to prevent generating excessively long or incompatible filenames.
MAX_TASK_STEM_BYTES = 200


class HybridDependencyError(RuntimeError):
    pass


def build_hybrid_dependency_error_message(backend: str) -> str:
    return (
        f"`{backend}` requires local pipeline dependencies (`mineru[pipeline]`, "
        "including `torch`). Install `mineru[pipeline]` or `mineru[core]`. "
        "If you need a lightweight remote client without local `torch`, "
        "use `vlm-http-client` instead."
    )


def ensure_backend_dependencies(backend: str) -> None:
    if not backend.startswith("hybrid-"):
        return
    if importlib.util.find_spec("torch") is None:
        raise HybridDependencyError(build_hybrid_dependency_error_message(backend))


def _load_hybrid_analyze_entrypoint(entrypoint_name: str, backend: str):
    ensure_backend_dependencies(backend)
    try:
        hybrid_analyze = importlib.import_module("mineru.backend.hybrid.hybrid_analyze")
    except (ImportError, ModuleNotFoundError) as exc:
        raise HybridDependencyError(
            build_hybrid_dependency_error_message(backend)
        ) from exc
    return getattr(hybrid_analyze, entrypoint_name)


def utf8_byte_length(value: str) -> int:
    return len(value.encode("utf-8"))


def truncate_to_utf8_bytes(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""

    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value

    truncated = encoded[:max_bytes]
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError as exc:
            truncated = truncated[:exc.start]
    return ""


def normalize_task_stem(stem: str, max_bytes: int = MAX_TASK_STEM_BYTES) -> str:
    return truncate_to_utf8_bytes(stem, max_bytes)


def normalize_upload_filename(upload_name: str) -> str:
    sanitized_name = Path(upload_name).name
    sanitized_path = Path(sanitized_name)
    normalized_stem = normalize_task_stem(sanitized_path.stem)
    return f"{normalized_stem}{sanitized_path.suffix}"


def build_task_stem_candidate(
    stem: str,
    suffix: str = "",
    max_bytes: int = MAX_TASK_STEM_BYTES,
) -> str:
    if utf8_byte_length(f"{stem}{suffix}") <= max_bytes:
        return f"{stem}{suffix}"
    suffix_bytes = utf8_byte_length(suffix)
    if suffix_bytes >= max_bytes:
        return truncate_to_utf8_bytes(suffix, max_bytes)
    return f"{truncate_to_utf8_bytes(stem, max_bytes - suffix_bytes)}{suffix}"


def uniquify_task_stems(
    stems: Sequence[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Assign task-local unique stems while preserving input order."""
    normalized_inputs = [normalize_task_stem(stem) for stem in stems]
    raw_keys = {stem.casefold() for stem in normalized_inputs}
    occurrence_counts: dict[str, int] = {}
    assigned_keys: set[str] = set()
    unique_stems: list[str] = []
    renamed: list[tuple[str, str]] = []

    for stem, normalized_stem in zip(stems, normalized_inputs):
        stem_base = normalized_stem or stem
        stem_key = stem_base.casefold()
        seen_count = occurrence_counts.get(stem_key, 0)
        occurrence_counts[stem_key] = seen_count + 1

        if seen_count == 0 and stem_key not in assigned_keys:
            effective_stem = stem_base
        else:
            suffix = seen_count + 1
            while True:
                candidate = build_task_stem_candidate(stem_base, f"_{suffix}")
                candidate_key = candidate.casefold()
                if candidate_key not in raw_keys and candidate_key not in assigned_keys:
                    effective_stem = candidate
                    break
                suffix += 1

        assigned_keys.add(effective_stem.casefold())
        unique_stems.append(effective_stem)
        if effective_stem != stem:
            renamed.append((stem, effective_stem))

    return unique_stems, renamed


def read_fn(path, file_suffix: str | None = None):
    if not isinstance(path, Path):
        path = Path(path)
    with open(str(path), "rb") as input_file:
        file_bytes = input_file.read()
        if file_suffix is None:
            file_suffix = guess_suffix_by_bytes(file_bytes, path)
        if file_suffix in image_suffixes:
            return images_bytes_to_pdf_bytes(file_bytes)
        elif file_suffix in pdf_suffixes + office_suffixes:
            return file_bytes
        else:
            raise Exception(f"Unknown file suffix: {file_suffix}")


def prepare_env(output_dir, pdf_file_name, parse_method):
    local_md_dir = str(os.path.join(output_dir, pdf_file_name, parse_method))
    local_image_dir = os.path.join(str(local_md_dir), "images")
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir


def _resolve_storage_backend(
    scope: str,
    storage_options=None,
    *,
    custom_config: CustomConfig | None = None,
) -> str:
    if (
        isinstance(storage_options, dict)
        and isinstance(storage_options.get(scope), dict)
        and storage_options[scope].get("backend")
    ):
        return str(storage_options[scope]["backend"]).strip().lower()
    if custom_config is not None:
        return "local"
    scoped_env = os.getenv(f"MINERU_STORAGE_{scope.upper()}_BACKEND")
    if scoped_env:
        return scoped_env.strip().lower()
    shared_env = os.getenv("MINERU_STORAGE_BACKEND")
    if shared_env:
        return shared_env.strip().lower()
    return "local"


def _join_image_ref(prefix: str, image_path: str) -> str:
    if not prefix:
        return image_path
    return f"{prefix.rstrip('/')}/{str(image_path).lstrip('/')}"


def _collect_uploaded_image_urls(pdf_info, image_prefix: str) -> list[str]:
    urls = set()

    def _walk(node):
        if isinstance(node, dict):
            image_path = node.get("image_path")
            if isinstance(image_path, str) and image_path.strip():
                urls.add(_join_image_ref(image_prefix, image_path))

            html = node.get("html")
            if isinstance(html, str) and html:
                for src in re.findall(r'src="(?!data:)([^"]+)"', html):
                    if src.strip():
                        urls.add(_join_image_ref(image_prefix, src))

            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(pdf_info)
    return sorted(urls)


def convert_pdf_bytes_to_bytes(pdf_bytes, start_page_id=0, end_page_id=None):
    try:
        rebuilt_pdf_bytes = rewrite_pdf_bytes_with_pdfium(
            pdf_bytes,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )
        if rebuilt_pdf_bytes:
            return rebuilt_pdf_bytes
        logger.warning("PDFium rewrite returned empty bytes, using original PDF bytes.")
    except Exception as fallback_error:
        logger.warning(
            f"Error in converting PDF bytes with pdfium: {fallback_error}, "
            "using original PDF bytes."
        )
    return pdf_bytes


def _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id):
    """准备处理PDF字节数据"""
    result = []
    for pdf_bytes in pdf_bytes_list:
        new_pdf_bytes = convert_pdf_bytes_to_bytes(pdf_bytes, start_page_id, end_page_id)
        result.append(new_pdf_bytes)
    return result


def _process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_orig_pdf,
        f_dump_md,
        f_dump_content_list,
        f_dump_middle_json,
        f_dump_model_output,
        f_make_md_mode,
        middle_json,
        model_output=None,
        process_mode="vlm",
        storage_options=None,
        f_dump_images_manifest: bool = True,
        custom_config: CustomConfig | None = None,
        output_md: str | None = None,
):
    """
    该函数负责根据不同的参数与模式，处理文档解析的输出结果，并将相关文件（如可视化PDF、Markdown、JSON等）生成或保存到相应目录下。

    主要功能包括：
    1. 根据 process_mode 选择对应的 Markdown 和内容列表生成函数。
    2. 可选绘制页面排版框/内容框，并将可视化结果保存为PDF。
    3. 保存原始PDF或office文件副本。
    4. 输出主Markdown文件、内容列表（JSON）、以及其他中间/模型输出（如设定）。
    5. 支持不同文档类型和生成模式，确保输出格式与内容完整。
    """
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
    if process_mode == "pipeline":
        make_func = pipeline_union_make
    elif process_mode == "vlm":
        make_func = vlm_union_make
    elif process_mode in office_suffixes:
        make_func = office_union_make
    else:
        raise Exception(f"Unknown process_mode: {process_mode}")
    """处理输出文件"""
    if f_draw_layout_bbox: # 在原始 PDF 上叠加段落级彩色 bbox 可视化
        try:
            draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")
        except Exception as exc:
            logger.warning(f"Skipping layout bbox visualization for {pdf_file_name}: {exc}")

    if f_draw_span_bbox: # 在原始 PDF 上叠加 span 级彩色 bbox 可视化
        try:
            draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")
        except Exception as exc:
            logger.warning(f"Skipping span bbox visualization for {pdf_file_name}: {exc}")

    if f_dump_orig_pdf: # 原始 PDF 副本（或 office 原文件）
        if process_mode in ["pipeline", "vlm"]:
            md_writer.write(
                f"{pdf_file_name}_origin.pdf",
                pdf_bytes,
            )
        elif process_mode in office_suffixes:
            md_writer.write(
                f"{pdf_file_name}_origin.{process_mode}",
                pdf_bytes,
            )

    image_dir = resolve_image_output_ref_prefix(local_image_dir, custom_config=custom_config)
    image_backend = _resolve_storage_backend("image", storage_options=storage_options, custom_config=custom_config)
    if image_backend != "local" and f_dump_images_manifest:
        image_urls = _collect_uploaded_image_urls(pdf_info, image_dir)
        md_writer.write_string(
            "images.json",
            json.dumps(image_urls, ensure_ascii=False, indent=2),
        )

    enhance_enable = resolve_enhancement_enabled(custom_config=custom_config)
    # 智能推断：output_md=None 时根据 enhance_enable 决定是否运行增强
    effective_output_md = output_md if output_md is not None else ("enhanced" if enhance_enable else "raw")
    run_enhancement = enhance_enable and effective_output_md in ("auto", "enhanced", "both")

    content_list_v2 = None
    if f_dump_md:
        md_content_str = make_func(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(f"{pdf_file_name}.md", md_content_str)  # 始终写出原始 Markdown
        if run_enhancement:
            content_list_v2 = make_func(pdf_info, MakeMode.CONTENT_LIST_V2, image_dir)
            enhancement_payload, enhanced_md = run_enhancement_pipeline(
                custom_config=custom_config,
                pdf_file_name=pdf_file_name,
                process_mode=process_mode,
                middle_json=middle_json,
                content_list_v2=content_list_v2,
                raw_markdown=md_content_str,
                image_dir=local_image_dir,
            )
            md_writer.write_string(
                f"{pdf_file_name}_enhance.json",
                json.dumps(enhancement_payload, ensure_ascii=False, indent=2),
            )
            md_writer.write_string(f"{pdf_file_name}_enhanced.md", enhanced_md)  # 增强版 Markdown

    if f_dump_content_list:

        content_list = make_func(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json", # 扁平内容列表 v1
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

        if content_list_v2 is None:
            content_list_v2 = make_func(pdf_info, MakeMode.CONTENT_LIST_V2, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list_v2.json", # 分页内容列表 v2
            json.dumps(content_list_v2, ensure_ascii=False, indent=4),
        )


    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json", # 结构化中间 JSON
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

    if f_dump_model_output:
        md_writer.write_string(
            f"{pdf_file_name}_model.json", # 模型原始输出
            json.dumps(model_output, ensure_ascii=False, indent=4),
        )

    logger.debug(f"local output dir is {local_md_dir}")


def _process_pipeline(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        p_lang_list,
        parse_method,
        p_formula_enable,
        p_table_enable,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
        discard_types=None,
        storage_options=None,
        custom_config: CustomConfig | None = None,
        output_md: str | None = None,
):
    """处理pipeline后端逻辑"""
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming as pipeline_doc_analyze_streaming

    image_writer_list = []
    md_writer_list = []
    local_output_info = []
    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        if _resolve_storage_backend("image", storage_options=storage_options, custom_config=custom_config) == "local":
            os.makedirs(local_image_dir, exist_ok=True)
        image_writer, md_writer = resolve_output_writers(
            local_image_dir,
            local_md_dir,
            custom_config=custom_config,
        )
        image_writer_list.append(image_writer)
        md_writer_list.append(md_writer)
        local_output_info.append((pdf_file_name, local_image_dir, local_md_dir))

    output_futures = []

    def run_output_task(doc_index, middle_json, model_list):
        pdf_file_name, local_image_dir, local_md_dir = local_output_info[doc_index]
        md_writer = md_writer_list[doc_index]
        pdf_bytes = pdf_bytes_list[doc_index]
        logger.debug(f"Pipeline output start: doc{doc_index}")
        try:
            _process_output(
                middle_json["pdf_info"], pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
                md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
                f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
                f_make_md_mode, middle_json, model_list, process_mode="pipeline",
                storage_options=storage_options,
                custom_config=custom_config,
                output_md=output_md,
            )
            logger.debug(f"Pipeline output complete: doc{doc_index}")
        except Exception:
            logger.exception(f"Pipeline output failed: doc{doc_index}")
            raise

    with ThreadPoolExecutor(max_workers=1) as output_executor:
        # on_doc_ready 回调的实现在上层调用方（如 MinerUPipeline），负责将 middle_json 序列化写入 middle.json 文件，将 model_list 写入 _model.json 文件
        def on_doc_ready(doc_index, model_list, middle_json, ocr_enable):
            logger.debug(
                f"Pipeline doc ready: doc{doc_index} pages={len(middle_json['pdf_info'])} output_submitted=1"
            )
            future = output_executor.submit(run_output_task, doc_index, middle_json, model_list)
            output_futures.append(future)

        pipeline_doc_analyze_streaming(
            pdf_bytes_list,
            image_writer_list,
            p_lang_list,
            on_doc_ready,
            parse_method=parse_method,
            formula_enable=p_formula_enable,
            table_enable=p_table_enable,
            discard_types=discard_types,
        )

        for future in output_futures:
            future.result()
    return


async def _async_process_vlm(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        backend,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
        discard_types=None,
        server_url=None,
        storage_options=None,
        custom_config: CustomConfig | None = None,
        output_md: str | None = None,
        **kwargs,
):
    """异步处理VLM后端逻辑"""
    parse_method = "vlm"
    f_draw_span_bbox = False
    if not backend.endswith("client"):
        server_url = None

    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        if _resolve_storage_backend("image", storage_options=storage_options, custom_config=custom_config) == "local":
            os.makedirs(local_image_dir, exist_ok=True)
        image_writer, md_writer = resolve_output_writers(
            local_image_dir,
            local_md_dir,
            custom_config=custom_config,
        )

        middle_json, infer_result = await aio_vlm_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend=backend,
            server_url=server_url,
            discard_types=discard_types,
            **kwargs,
        )

        pdf_info = middle_json["pdf_info"]

        _process_output(
            pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
            md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
            f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
            f_make_md_mode, middle_json, infer_result, process_mode="vlm",
            storage_options=storage_options,
            custom_config=custom_config,
            output_md=output_md,
        )


def _process_vlm(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        backend,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
        discard_types=None,
        server_url=None,
        storage_options=None,
        custom_config: CustomConfig | None = None,
        output_md: str | None = None,
        **kwargs,
):
    """同步处理VLM后端逻辑"""
    parse_method = "vlm"
    f_draw_span_bbox = False
    if not backend.endswith("client"):
        server_url = None

    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        if _resolve_storage_backend("image", storage_options=storage_options, custom_config=custom_config) == "local":
            os.makedirs(local_image_dir, exist_ok=True)
        image_writer, md_writer = resolve_output_writers(
            local_image_dir,
            local_md_dir,
            custom_config=custom_config,
        )

        middle_json, infer_result = vlm_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend=backend,
            server_url=server_url,
            discard_types=discard_types,
            **kwargs,
        )

        pdf_info = middle_json["pdf_info"]

        _process_output(
            pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
            md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
            f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
            f_make_md_mode, middle_json, infer_result, process_mode="vlm",
            storage_options=storage_options,
            custom_config=custom_config,
            output_md=output_md,
        )


def _process_hybrid(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        h_lang_list,
        parse_method,
        inline_formula_enable,
        backend,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
        discard_types=None,
        server_url=None,
        storage_options=None,
        custom_config: CustomConfig | None = None,
        output_md: str | None = None,
        **kwargs,
):
    hybrid_doc_analyze = _load_hybrid_analyze_entrypoint(
        "doc_analyze",
        f"hybrid-{backend}",
    )
    """同步处理hybrid后端逻辑"""
    if not backend.endswith("client"):
        server_url = None

    for idx, (pdf_bytes, lang) in enumerate(zip(pdf_bytes_list, h_lang_list)):
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, f"hybrid_{parse_method}")
        if _resolve_storage_backend("image", storage_options=storage_options, custom_config=custom_config) == "local":
            os.makedirs(local_image_dir, exist_ok=True)
        image_writer, md_writer = resolve_output_writers(
            local_image_dir,
            local_md_dir,
            custom_config=custom_config,
        )

        middle_json, infer_result, _vlm_ocr_enable = hybrid_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend=backend,
            parse_method=parse_method,
            language=lang,
            inline_formula_enable=inline_formula_enable,
            discard_types=discard_types,
            server_url=server_url,
            **kwargs,
        )

        pdf_info = middle_json["pdf_info"]

        # f_draw_span_bbox = not _vlm_ocr_enable
        f_draw_span_bbox = False

        _process_output(
            pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
            md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
            f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
            f_make_md_mode, middle_json, infer_result, process_mode="vlm",
            storage_options=storage_options,
            custom_config=custom_config,
            output_md=output_md,
        )


async def _async_process_hybrid(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        h_lang_list,
        parse_method,
        inline_formula_enable,
        backend,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
        discard_types=None,
        server_url=None,
        storage_options=None,
        custom_config: CustomConfig | None = None,
        output_md: str | None = None,
        **kwargs,
):
    aio_hybrid_doc_analyze = _load_hybrid_analyze_entrypoint(
        "aio_doc_analyze",
        f"hybrid-{backend}",
    )
    """异步处理hybrid后端逻辑"""
    if not backend.endswith("client"):
        server_url = None

    for idx, (pdf_bytes, lang) in enumerate(zip(pdf_bytes_list, h_lang_list)):
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, f"hybrid_{parse_method}")
        if _resolve_storage_backend("image", storage_options=storage_options, custom_config=custom_config) == "local":
            os.makedirs(local_image_dir, exist_ok=True)
        image_writer, md_writer = resolve_output_writers(
            local_image_dir,
            local_md_dir,
            custom_config=custom_config,
        )

        middle_json, infer_result, _vlm_ocr_enable = await aio_hybrid_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend=backend,
            parse_method=parse_method,
            language=lang,
            inline_formula_enable=inline_formula_enable,
            discard_types=discard_types,
            server_url=server_url,
            **kwargs,
        )

        pdf_info = middle_json["pdf_info"]

        # f_draw_span_bbox = not _vlm_ocr_enable
        f_draw_span_bbox = False

        _process_output(
            pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
            md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
            f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
            f_make_md_mode, middle_json, infer_result, process_mode="vlm",
            storage_options=storage_options,
            custom_config=custom_config,
            output_md=output_md,
        )


def _process_office_doc(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_file=True,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
        storage_options=None,
        custom_config: CustomConfig | None = None,
        output_md: str | None = None,
):
    need_remove_index = []
    for i, file_bytes in enumerate(pdf_bytes_list):
        pdf_file_name = pdf_file_names[i]
        file_suffix = guess_suffix_by_bytes(file_bytes)
        if file_suffix in docx_suffixes:

            need_remove_index.append(i)

            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, f"office")
            if _resolve_storage_backend("image", storage_options=storage_options, custom_config=custom_config) == "local":
                os.makedirs(local_image_dir, exist_ok=True)
            image_writer, md_writer = resolve_output_writers(
                local_image_dir,
                local_md_dir,
                custom_config=custom_config,
            )
            middle_json, infer_result = office_docx_analyze(
                file_bytes,
                image_writer=image_writer,
                discard_policy=resolve_discard_policy(
                    (custom_config.discard or {}).get("types") if custom_config else None
                ),
            )

            f_draw_layout_bbox = False
            f_draw_span_bbox = False
            pdf_info = middle_json["pdf_info"]

            _process_output(
                pdf_info, file_bytes, pdf_file_name, local_md_dir, local_image_dir,
                md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_file,
                f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
                f_make_md_mode, middle_json, infer_result, process_mode="docx",
                storage_options=storage_options,
                custom_config=custom_config,
                output_md=output_md,
            )
        elif file_suffix in pptx_suffixes:
            need_remove_index.append(i)
            logger.warning(f"Currently, PPTX files are not supported: {pdf_file_name}")
        elif file_suffix in xlsx_suffixes:
            need_remove_index.append(i)
            logger.warning(f"Currently, XLSX files are not supported: {pdf_file_name}")

    return need_remove_index


def do_parse(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        p_lang_list: list[str],
        backend="pipeline",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        server_url=None,
        f_draw_layout_bbox=True,
        f_draw_span_bbox=True,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_pdf=True,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
        start_page_id=0,
        end_page_id=None,
        discard_types=None,
        custom_config: CustomConfig | None = None,
        output_md: str | None = None,
        **kwargs,
):
    storage_options = kwargs.pop("storage_options", None)
    if custom_config is not None:
        discard_node = custom_config.discard or {}
        discard_types = discard_node.get("types")
        storage_options = custom_config.storage or {}
    need_remove_index = _process_office_doc(
        output_dir,
        pdf_file_names=pdf_file_names,
        pdf_bytes_list=pdf_bytes_list,
        f_dump_md=f_dump_md,
        f_dump_middle_json=f_dump_middle_json,
        f_dump_model_output=f_dump_model_output,
        f_dump_orig_file=f_dump_orig_pdf,
        f_dump_content_list=f_dump_content_list,
        f_make_md_mode=f_make_md_mode,
        storage_options=storage_options,
        custom_config=custom_config,
        output_md=output_md,
    )
    for index in sorted(need_remove_index, reverse=True):
        del pdf_bytes_list[index]
        del pdf_file_names[index]
        del p_lang_list[index]
    if not pdf_bytes_list:
        logger.warning("No valid PDF or image files to process.")
        return

    # 预处理PDF字节数据
    pdf_bytes_list = _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id)

    if backend == "pipeline":
        _process_pipeline(
            output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
            parse_method, formula_enable, table_enable,
            f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
            f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
            discard_types=discard_types,
            storage_options=storage_options,
            custom_config=custom_config,
            output_md=output_md,
        )
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]

            if backend == "vllm-async-engine":
                raise Exception("vlm-vllm-async-engine backend is not supported in sync mode, please use vlm-vllm-engine backend")

            if backend == "auto-engine":
                backend = get_vlm_engine(inference_engine='auto', is_async=False)

            os.environ['MINERU_VLM_FORMULA_ENABLE'] = str(formula_enable)
            os.environ['MINERU_VLM_TABLE_ENABLE'] = str(table_enable)

            _process_vlm(
                output_dir, pdf_file_names, pdf_bytes_list, backend,
                f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
                f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
                discard_types=discard_types, server_url=server_url,
                storage_options=storage_options, custom_config=custom_config,
                output_md=output_md, **kwargs,
            )
        elif backend.startswith("hybrid-"):
            ensure_backend_dependencies(backend)
            backend = backend[7:]

            if backend == "vllm-async-engine":
                raise Exception(
                    "hybrid-vllm-async-engine backend is not supported in sync mode, please use hybrid-vllm-engine backend")

            if backend == "auto-engine":
                backend = get_vlm_engine(inference_engine='auto', is_async=False)

            os.environ['MINERU_VLM_TABLE_ENABLE'] = str(table_enable)
            os.environ['MINERU_VLM_FORMULA_ENABLE'] = "true"

            _process_hybrid(
                output_dir, pdf_file_names, pdf_bytes_list, p_lang_list, parse_method, formula_enable, backend,
                f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
                f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
                discard_types=discard_types, server_url=server_url,
                storage_options=storage_options, custom_config=custom_config,
                output_md=output_md, **kwargs,
            )


async def aio_do_parse(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        p_lang_list: list[str],
        backend="pipeline",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        server_url=None,
        f_draw_layout_bbox=True,
        f_draw_span_bbox=True,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_pdf=True,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
        start_page_id=0,
        end_page_id=None,
        discard_types=None,
        custom_config: CustomConfig | None = None,
        output_md: str | None = None,
        **kwargs,
):
    storage_options = kwargs.pop("storage_options", None)
    if custom_config is not None:
        discard_node = custom_config.discard or {}
        discard_types = discard_node.get("types")
        storage_options = custom_config.storage or {}
    need_remove_index = _process_office_doc(
        output_dir,
        pdf_file_names=pdf_file_names,
        pdf_bytes_list=pdf_bytes_list,
        f_dump_md=f_dump_md,
        f_dump_middle_json=f_dump_middle_json,
        f_dump_model_output=f_dump_model_output,
        f_dump_orig_file=f_dump_orig_pdf,
        f_dump_content_list=f_dump_content_list,
        f_make_md_mode=f_make_md_mode,
        storage_options=storage_options,
        custom_config=custom_config,
        output_md=output_md,
    )
    for index in sorted(need_remove_index, reverse=True):
        del pdf_bytes_list[index]
        del pdf_file_names[index]
        del p_lang_list[index]
    if not pdf_bytes_list:
        logger.warning("No valid PDF or image files to process.")
        return

    # 预处理PDF字节数据
    pdf_bytes_list = _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id)

    if backend == "pipeline":
        # pipeline模式暂不支持异步，使用同步处理方式
        _process_pipeline(
            output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
            parse_method, formula_enable, table_enable,
            f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
            f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
            discard_types=discard_types,
            storage_options=storage_options,
            custom_config=custom_config,
            output_md=output_md,
        )
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]

            if backend == "vllm-engine":
                raise Exception("vlm-vllm-engine backend is not supported in async mode, please use vlm-vllm-async-engine backend")

            if backend == "auto-engine":
                backend = get_vlm_engine(inference_engine='auto', is_async=True)

            os.environ['MINERU_VLM_FORMULA_ENABLE'] = str(formula_enable)
            os.environ['MINERU_VLM_TABLE_ENABLE'] = str(table_enable)

            await _async_process_vlm(
                output_dir, pdf_file_names, pdf_bytes_list, backend,
                f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
                f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
                discard_types=discard_types, server_url=server_url,
                storage_options=storage_options, custom_config=custom_config,
                output_md=output_md, **kwargs,
            )
        elif backend.startswith("hybrid-"):
            ensure_backend_dependencies(backend)
            backend = backend[7:]

            if backend == "vllm-engine":
                raise Exception("hybrid-vllm-engine backend is not supported in async mode, please use hybrid-vllm-async-engine backend")

            if backend == "auto-engine":
                backend = get_vlm_engine(inference_engine='auto', is_async=True)

            os.environ['MINERU_VLM_TABLE_ENABLE'] = str(table_enable)
            os.environ['MINERU_VLM_FORMULA_ENABLE'] = "true"

            await _async_process_hybrid(
                output_dir, pdf_file_names, pdf_bytes_list, p_lang_list, parse_method, formula_enable, backend,
                f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
                f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
                discard_types=discard_types, server_url=server_url,
                storage_options=storage_options, custom_config=custom_config,
                output_md=output_md, **kwargs,
            )


if __name__ == "__main__":
    # pdf_path = "../../demo/pdfs/demo3.pdf"
    pdf_path = "C:/Users/zhaoxiaomeng/Downloads/4546d0e2-ba60-40a5-a17e-b68555cec741.pdf"

    try:
       do_parse("./output", [Path(pdf_path).stem], [read_fn(Path(pdf_path))],["ch"],
                end_page_id=10,
                backend='vlm-huggingface'
                # backend = 'pipeline'
                )
    except Exception as e:
        logger.exception(e)
