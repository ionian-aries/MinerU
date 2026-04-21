# Copyright (c) Opendatalab. All rights reserved.
import base64
import copy
import os
import re
import time

from loguru import logger
from tqdm import tqdm

from mineru.backend.utils import cross_page_table_merge
from mineru.utils.config_reader import get_device, get_llm_aided_config, get_formula_enable
from mineru.backend.pipeline.model_init import AtomModelSingleton
from mineru.backend.pipeline.para_split import para_split
from mineru.utils.char_utils import full_to_half
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType, BlockType
from mineru.utils.llm_aided import llm_aided_title
from mineru.utils.model_utils import clean_memory
from mineru.backend.pipeline.pipeline_magic_model import MagicModel
from mineru.utils.ocr_utils import OcrConfidence, rotate_vertical_crop_if_needed
from mineru.version import __version__
from mineru.utils.hash_utils import bytes_md5, str_sha256
from mineru.utils.pdfium_guard import close_pdfium_document, pdfium_guard


def _save_base64_image(b64_data_uri: str, image_writer, page_index: int):
    """Persist a data-URI image via image_writer and return a relative path."""
    m = re.match(r'data:image/(\w+);base64,(.+)', b64_data_uri, re.DOTALL)
    if not m:
        logger.warning(f"Unrecognized image_base64 format in page {page_index}, skipping.")
        return None

    fmt = m.group(1)
    ext = "jpg" if fmt == "jpeg" else fmt
    try:
        img_bytes = base64.b64decode(m.group(2))
    except Exception as e:
        logger.warning(f"Failed to decode image_base64 on page {page_index}: {e}")
        return None

    img_path = f"{str_sha256(b64_data_uri)}.{ext}"
    image_writer.write(img_path, img_bytes)
    return img_path


def _replace_inline_base64_img_src(markup: str, image_writer, page_index: int) -> str:
    """Replace inline base64 img src attributes with local relative paths."""
    if not markup or "base64," not in markup:
        return markup

    def _replace_src(match, _writer=image_writer, _idx=page_index): # 正则匹配 src="data:image/...;base64,..."
        img_path = _save_base64_image(match.group(1), _writer, _idx) # base64 解码 → {sha256}.{ext} 文件 → 替换 HTML src 为本地路径
        if img_path:
            return f'src="{img_path}"'
        return match.group(0)

    return re.sub(
        r'src="(data:image/[^"]+)"',
        _replace_src,
        markup,
    )


def _replace_inline_table_images(preproc_blocks: list[dict], image_writer, page_index: int) -> None:
    """Persist inline base64 images embedded inside table HTML."""
    # 遍历 TABLE → TABLE_BODY → span.html
    if not image_writer:
        return

    for block in preproc_blocks:
        if block.get("type") != BlockType.TABLE:
            continue

        for sub_block in block.get("blocks", []):
            if sub_block.get("type") != BlockType.TABLE_BODY:
                continue

            for line in sub_block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("type") != ContentType.TABLE:
                        continue
                    span["html"] = _replace_inline_base64_img_src(
                        span.get("html", ""),
                        image_writer,
                        page_index,
                    )

# 单页转换的核心函数
def page_model_info_to_page_info(
    page_model_info,
    image_dict,
    page,
    image_writer,
    page_index,
    ocr_enable=False,
    discard_policy=None,
):
    scale = image_dict["scale"] # 页面缩放比例
    page_pil_img = image_dict["img_pil"] # 页面 PIL 图片
    page_img_md5 = bytes_md5(page_pil_img.tobytes()) # 图片 MD5（用于图片文件命名）
    with pdfium_guard():
        page_w, page_h = map(int, page.get_size()) # PDF 原始页面尺寸
    # MagicModel 构造，__init__ 内部顺序执行 7 个子步骤
    magic_model = MagicModel(
        page_model_info,
        page,
        scale,
        page_pil_img,
        page_w,
        page_h,
        ocr_enable,
        discard_policy=discard_policy,
    )

    """从magic_model对象中获取后面会用到的区块信息"""
    preproc_blocks = magic_model.get_preproc_blocks() # 主要内容块
    discarded_blocks = magic_model.get_discarded_blocks() # 页眉/页脚/页码等
    all_image_spans = magic_model.get_all_image_spans() # 需要截图的 span

    # 对image/table/chart/interline_equation的span截图
    for span in all_image_spans:
        if span["type"] in [
            ContentType.IMAGE,
            ContentType.TABLE,
            ContentType.CHART,
            ContentType.SEAL,
            ContentType.INTERLINE_EQUATION
        ]:
            # 从页面 PIL 图片按 bbox 裁剪，写入 images/{type}/{md5}_{page_id}_{idx}.jpg，将路径存入 span["image_path"]
            span = cut_image_and_table(span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale)

    """构造page_info"""
    _replace_inline_table_images(preproc_blocks, image_writer, page_index) # 表格内嵌图片持久化

    page_info = make_page_info_dict(preproc_blocks, page_index, page_w, page_h, discarded_blocks) # 构造 page_info

    return page_info


def build_page_model_info(page_layout_dets, page_index, pil_img):
    page_info_dict = {'page_no': page_index, 'width': pil_img.width, 'height': pil_img.height}
    return {'layout_dets': page_layout_dets, 'page_info': page_info_dict}


def append_page_model_infos_to_middle_json(
    middle_json,
    page_model_infos,
    images_list,
    pdf_doc,
    image_writer,
    page_start_index=0,
    ocr_enable=False,
    discard_policy=None,
    progress_bar=None,
):
    # 逐页取出 PDF page 对象
    for offset, (page_model_info, image_dict) in enumerate(zip(page_model_infos, images_list)):
        page_index = page_start_index + offset
        with pdfium_guard():
            page = pdf_doc[page_index] # 取出 pdfium page 对象
        # 核心转换，生成的 page_info 被追加到 middle_json["pdf_info"] 列表中。
        page_info = page_model_info_to_page_info(
            copy.deepcopy(page_model_info),  # ★深拷贝，避免修改 model_list 中的原始数据
            image_dict,
            page,
            image_writer,
            page_index,
            ocr_enable=ocr_enable,
            discard_policy=discard_policy,
        )
        if page_info is None: # 空页兜底处理
            with pdfium_guard():
                page_w, page_h = map(int, pdf_doc[page_index].get_size())
            page_info = make_page_info_dict([], page_index, page_w, page_h, []) # ★追加到 middle_json
        middle_json["pdf_info"].append(page_info)
        if progress_bar is not None:
            progress_bar.update(1) # 更新进度条


def append_batch_results_to_middle_json(
    middle_json,
    batch_results,
    images_list,
    pdf_doc,
    image_writer,
    page_start_index=0,
    ocr_enable=False,
    discard_policy=None,
    model_list=None,
    progress_bar=None,
):  
    # 包装 page_model_info
    page_model_infos = []
    # 将扁平 layout元素包装为： {layout_dets: [...], page_info: {page_no, width, height}}
    for offset, (image_dict, page_layout_dets) in enumerate(zip(images_list, batch_results)):
        page_index = page_start_index + offset
        # 将扁平的 page_layout_dets 包装为 {layout_dets: [...], page_info: {page_no, width, height}}
        page_model_info = build_page_model_info(page_layout_dets, page_index, image_dict['img_pil'])
        page_model_infos.append(page_model_info)

    if model_list is not None:
        model_list.extend(page_model_infos) # 将 page_model_infos 追加到 model_list（最终写入 _model.json）

    # 逐页处理 append_page_model_infos_to_middle_json
    append_page_model_infos_to_middle_json(
        middle_json,
        page_model_infos,
        images_list,
        pdf_doc,
        image_writer,
        page_start_index=page_start_index,
        ocr_enable=ocr_enable,
        discard_policy=discard_policy,
        progress_bar=progress_bar,
    )


def _extract_text_from_block(block):
    text_parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("type") == ContentType.TEXT:
                text_parts.append(span.get("content", ""))
    return "".join(text_parts).strip()


def _iter_block_spans(block):
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            yield span

    for sub_block in block.get("blocks", []):
        yield from _iter_block_spans(sub_block)


def _normalize_formula_tag_content(tag_content):
    tag_content = full_to_half(tag_content.strip())
    if tag_content.startswith("("):
        tag_content = tag_content[1:].strip()
    if tag_content.endswith(")"):
        tag_content = tag_content[:-1].strip()
    return tag_content


def _get_interline_equation_span(block):
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("type") == ContentType.INTERLINE_EQUATION:
                return span
    return None


def _append_formula_number_tag(equation_block, formula_number_block):
    equation_span = _get_interline_equation_span(equation_block)
    tag_content = _normalize_formula_tag_content(_extract_text_from_block(formula_number_block))
    if equation_span is not None:
        formula = equation_span.get("content", "")
        equation_span["content"] = f"{formula}\\tag{{{tag_content}}}"


def _optimize_formula_number_blocks(pdf_info_list):
    for page_info in pdf_info_list:
        optimized_blocks = []
        blocks = page_info.get("preproc_blocks", [])
        for index, block in enumerate(blocks):
            if block.get("type") != BlockType.FORMULA_NUMBER:
                optimized_blocks.append(block)
                continue

            prev_block = blocks[index - 1] if index > 0 else None
            if prev_block and prev_block.get("type") == BlockType.INTERLINE_EQUATION:
                _append_formula_number_tag(prev_block, block)
                continue

            next_block = blocks[index + 1] if index + 1 < len(blocks) else None
            next_next_block = blocks[index + 2] if index + 2 < len(blocks) else None
            if (
                next_block
                and next_block.get("type") == BlockType.INTERLINE_EQUATION
                and (next_next_block is None or next_next_block.get("type") != BlockType.FORMULA_NUMBER)
            ):
                _append_formula_number_tag(next_block, block)
                continue

            block["type"] = BlockType.TEXT
            optimized_blocks.append(block)

        page_info["preproc_blocks"] = optimized_blocks


def _apply_post_ocr(pdf_info_list, lang=None):
    need_ocr_list = []
    img_crop_list = []

    for page_info in pdf_info_list:
        for block in page_info.get('preproc_blocks', []):
            for span in _iter_block_spans(block):
                if 'np_img' in span:
                    need_ocr_list.append(span)
                    # Keep post-OCR rec aligned with the main OCR pipeline for vertical tall crops.
                    img_crop_list.append(rotate_vertical_crop_if_needed(span['np_img']))
                    span.pop('np_img')

        for block in page_info.get('discarded_blocks', []):
            for span in _iter_block_spans(block):
                if 'np_img' in span:
                    need_ocr_list.append(span)
                    # Keep post-OCR rec aligned with the main OCR pipeline for vertical tall crops.
                    img_crop_list.append(rotate_vertical_crop_if_needed(span['np_img']))
                    span.pop('np_img')

    if len(img_crop_list) == 0:
        return

    atom_model_manager = AtomModelSingleton()
    ocr_model = atom_model_manager.get_atom_model(
        atom_model_name='ocr',
        det_db_box_thresh=0.3,
        lang=lang
    )
    ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]
    assert len(ocr_res_list) == len(
        need_ocr_list), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_list)}'
    for index, span in enumerate(need_ocr_list):
        ocr_text, ocr_score = ocr_res_list[index]
        if ocr_score > OcrConfidence.min_confidence:
            span['content'] = ocr_text
            span['score'] = float(f"{ocr_score:.3f}")
        else:
            span['content'] = ''
            span['score'] = 0.0


def _post_block_process(pdf_info_list):
    for page_info in pdf_info_list:
        for block_key in ["preproc_blocks", "para_blocks"]:
            for block in page_info.get(block_key, []):
                block_type = block.get("type")
                if block_type == BlockType.DOC_TITLE:
                    block["type"] = BlockType.TITLE
                    block["level"] = 1
                elif block_type == BlockType.PARAGRAPH_TITLE:
                    block["type"] = BlockType.TITLE
                    block["level"] = 2
                elif block_type == BlockType.VERTICAL_TEXT:
                    block["type"] = BlockType.TEXT


def finalize_middle_json(pdf_info_list, lang=None, ocr_enable=False):
    """Apply document-level post processing once all page_info entries are ready."""
    _apply_post_ocr(pdf_info_list, lang=lang) # ① 对含 np_img 的 span 执行补充 OCR（仅识别，不检测），填充 content 和 score
    _optimize_formula_number_blocks(pdf_info_list) # ② 将公式编号块（FORMULA_NUMBER）合并到相邻公式块的 LaTeX 里，用 \tag{} 包裹
    para_split(pdf_info_list) # ③ 对文本块进行段落拆分（基于行尾标点、缩进、行距等启发式规则），生成 para_blocks
    cross_page_table_merge(pdf_info_list) # ④ 检测并合并跨页表格（由 MINERU_TABLE_MERGE_ENABLE 环境变量控制）

    llm_aided_config = get_llm_aided_config() # ⑤ 用 LLM 辅助重新判断标题层级（可选）
    if llm_aided_config is not None:
        title_aided_config = llm_aided_config.get('title_aided', None)
        if title_aided_config is not None and title_aided_config.get('enable', False):
            llm_aided_title_start_time = time.time()
            llm_aided_title(pdf_info_list, title_aided_config)
            logger.info(f'llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}')

    _post_block_process(pdf_info_list) # ⑥ 块类型规范化：DOC_TITLE → TITLE(level=1)、PARAGRAPH_TITLE → TITLE(level=2)、VERTICAL_TEXT → TEXT

    if os.getenv('MINERU_DONOT_CLEAN_MEM') is None and len(pdf_info_list) >= 10:
        clean_memory(get_device()) # ⑦ 清理显存


def init_middle_json():
    return {"pdf_info": [], "_backend": "pipeline", "_version_name": __version__}


def result_to_middle_json(model_list, images_list, pdf_doc, image_writer, lang=None, ocr_enable=False, formula_enable=None):
    middle_json = init_middle_json()
    with tqdm(total=len(model_list), desc="Processing pages") as progress_bar:
        append_page_model_infos_to_middle_json(
            middle_json,
            model_list,
            images_list,
            pdf_doc,
            image_writer,
            ocr_enable=ocr_enable,
            progress_bar=progress_bar,
        )

    finalize_middle_json(middle_json["pdf_info"], lang=lang, ocr_enable=ocr_enable)
    close_pdfium_document(pdf_doc)
    return middle_json


def make_page_info_dict(blocks, page_id, page_w, page_h, discarded_blocks):
    return_dict = {
        'preproc_blocks': blocks, # 主内容块列表
        'page_idx': page_id, # 页码
        'page_size': [page_w, page_h], # 页面尺寸
        'discarded_blocks': discarded_blocks, # 被丢弃的块
    }
    return return_dict
