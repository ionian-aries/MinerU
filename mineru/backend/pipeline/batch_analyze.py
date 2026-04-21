import base64
import html

import cv2
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from .model_init import AtomModelSingleton
from .model_list import AtomicModel
from ...utils.config_reader import (
    get_formula_enable,
    get_ocr_det_mask_inline_formula_enable,
    get_table_enable,
)
from ...utils.bbox_utils import normalize_to_int_bbox
from ...utils.model_utils import crop_img, get_res_list_from_layout_res, clean_vram
from ...utils.ocr_utils import merge_det_boxes, update_det_boxes, sorted_boxes
from ...utils.ocr_utils import (
    get_adjusted_mfdetrec_res,
    get_ocr_result_list,
    OcrConfidence,
    get_rotate_crop_image_for_text_rec,
)
from ...utils.pdf_image_tools import get_crop_np_img

LAYOUT_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 8
TABLE_ORI_CLS_BATCH_SIZE = 16
TABLE_Wired_Wireless_CLS_BATCH_SIZE = 16


class BatchAnalyze:
    def __init__(
        self,
        model_manager,
        batch_ratio: int,
        formula_enable,
        table_enable,
        enable_ocr_det_batch: bool = True,
        table_ori_cls_batch_enabled: bool | None = None,
        text_ocr_det_batch_enabled: bool | None = None,
        mask_inline_formula_for_ocr_det: bool = True,
    ):
        self.batch_ratio = batch_ratio
        self.formula_enable = get_formula_enable(formula_enable)
        self.table_enable = get_table_enable(table_enable)
        self.model_manager = model_manager
        self.enable_ocr_det_batch = enable_ocr_det_batch
        self.table_ori_cls_batch_enabled = (
            enable_ocr_det_batch if table_ori_cls_batch_enabled is None else table_ori_cls_batch_enabled
        )
        self.text_ocr_det_batch_enabled = (
            enable_ocr_det_batch if text_ocr_det_batch_enabled is None else text_ocr_det_batch_enabled
        )
        self.mask_inline_formula_for_ocr_det = (
            get_ocr_det_mask_inline_formula_enable(mask_inline_formula_for_ocr_det)
        )

    @staticmethod
    def _apply_mask_boxes_to_image(
        bgr_image: np.ndarray,
        mask_boxes: list[dict] | None,
    ) -> np.ndarray:
        if not mask_boxes:
            return bgr_image

        masked_image = bgr_image.copy()
        image_h, image_w = masked_image.shape[:2]
        for mask_box in mask_boxes:
            bbox = mask_box.get("bbox")
            if bbox is None:
                continue

            int_bbox = normalize_to_int_bbox(bbox, image_size=(image_h, image_w))
            if int_bbox is None:
                continue

            x0, y0, x1, y1 = int_bbox
            masked_image[y0:y1, x0:x1] = 255

        return masked_image

    def _get_masked_det_image(
        self,
        bgr_image: np.ndarray,
        mask_boxes: list[dict] | None,
    ) -> np.ndarray:
        if not self.mask_inline_formula_for_ocr_det:
            return bgr_image
        return self._apply_mask_boxes_to_image(bgr_image, mask_boxes)

    @staticmethod
    def _prune_empty_ocr_text_blocks(layout_res: list[dict], ocr_enable: bool) -> None:
        if not ocr_enable or not layout_res:
            return

        def keep_item(item: dict) -> bool:
            if item.get("label") != "ocr_text":
                return True

            text = item.get("text")
            if isinstance(text, str):
                return bool(text.strip())
            return bool(text)

        layout_res[:] = [item for item in layout_res if keep_item(item)]

    @staticmethod
    def _bbox_center(bbox: list[float]) -> tuple[float, float]:
        return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _is_point_in_bbox(point: tuple[float, float], bbox: list[float]) -> bool:
        x, y = point
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

    @staticmethod
    def _bbox_intersection(bbox1: list[float], bbox2: list[float]) -> list[float] | None:
        x0 = max(float(bbox1[0]), float(bbox2[0]))
        y0 = max(float(bbox1[1]), float(bbox2[1]))
        x1 = min(float(bbox1[2]), float(bbox2[2]))
        y1 = min(float(bbox1[3]), float(bbox2[3]))
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]

    @classmethod
    def _bbox_intersection_area(cls, bbox1: list[float], bbox2: list[float]) -> float:
        overlap_bbox = cls._bbox_intersection(bbox1, bbox2)
        if overlap_bbox is None:
            return 0.0
        return float(overlap_bbox[2] - overlap_bbox[0]) * float(overlap_bbox[3] - overlap_bbox[1])

    @staticmethod
    def _bbox_to_relative_bbox(bbox: list[float], base_bbox: list[float]) -> list[float]:
        return [
            float(bbox[0]) - float(base_bbox[0]),
            float(bbox[1]) - float(base_bbox[1]),
            float(bbox[2]) - float(base_bbox[0]),
            float(bbox[3]) - float(base_bbox[1]),
        ]

    @staticmethod
    def _bbox_to_quad(bbox: list[float]) -> np.ndarray:
        x0, y0, x1, y1 = bbox
        return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

    @staticmethod
    def _encode_table_inline_image(np_img: np.ndarray, bbox: list[float]) -> str:
        image_h, image_w = np_img.shape[:2]
        image_bbox = normalize_to_int_bbox(bbox, image_size=(image_h, image_w))
        if image_bbox is None:
            return ""

        x0, y0, x1, y1 = image_bbox
        if x1 <= x0 or y1 <= y0:
            return ""

        crop_rgb = np_img[y0:y1, x0:x1]
        if crop_rgb.size == 0:
            return ""

        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".jpg", crop_bgr)
        if not success:
            return ""

        b64_str = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpg;base64,{b64_str}"

    @staticmethod
    def _get_virtual_image_bbox(bbox: list[float], box_size: float = 10.0) -> list[float]:
        center_x, center_y = BatchAnalyze._bbox_center(bbox)
        half_size = box_size / 2.0
        return [
            center_x - half_size,
            center_y - half_size,
            center_x + half_size,
            center_y + half_size,
        ]

    @staticmethod
    def _table_supports_inline_objects(table_res_dict: dict) -> bool:
        return str(table_res_dict.get("rotate_label", "0")) == "0"

    @staticmethod
    def _sort_table_ocr_result(ocr_result: list[list]) -> None:
        if not ocr_result:
            return

        sorted_result = sorted(
            ocr_result,
            key=lambda item: (float(np.asarray(item[0])[0][1]), float(np.asarray(item[0])[0][0])),
        )

        for i in range(len(sorted_result) - 1):
            for j in range(i, -1, -1):
                cur_box = np.asarray(sorted_result[j][0], dtype=np.float32)
                next_box = np.asarray(sorted_result[j + 1][0], dtype=np.float32)
                if (
                    abs(float(next_box[0][1]) - float(cur_box[0][1])) < 10
                    and float(next_box[0][0]) < float(cur_box[0][0])
                ):
                    sorted_result[j], sorted_result[j + 1] = sorted_result[j + 1], sorted_result[j]
                else:
                    break

        ocr_result[:] = sorted_result

    @classmethod
    def _extract_table_inline_objects(
        cls,
        layout_res: list[dict],
        np_img: np.ndarray,
        formula_enable: bool,
    ) -> dict[int, list[dict]]:
        image_h, image_w = np_img.shape[:2]
        image_size = (image_h, image_w)

        tables = []
        for res in layout_res:
            if res.get("label") != "table":
                continue
            table_bbox = normalize_to_int_bbox(res.get("bbox"), image_size=image_size)
            if table_bbox is None:
                continue
            tables.append((res, table_bbox))

        if not tables:
            return {}

        table_inline_objects = {id(table_res): [] for table_res, _ in tables}
        remove_ids = set()
        candidate_labels = {"image"}
        if formula_enable:
            candidate_labels.update({"inline_formula", "display_formula"})

        for layout_item in layout_res:
            label = layout_item.get("label")
            if label not in candidate_labels:
                continue

            item_bbox = normalize_to_int_bbox(layout_item.get("bbox"), image_size=image_size)
            if item_bbox is None:
                continue

            item_center = cls._bbox_center(item_bbox)
            matched_tables = []
            for table_res, table_bbox in tables:
                if not cls._is_point_in_bbox(item_center, table_bbox):
                    continue
                overlap_area = cls._bbox_intersection_area(item_bbox, table_bbox)
                matched_tables.append((overlap_area, table_res, table_bbox))

            if not matched_tables:
                continue

            matched_tables.sort(key=lambda item: item[0], reverse=True)
            _, table_res, table_bbox = matched_tables[0]
            overlap_bbox = cls._bbox_intersection(item_bbox, table_bbox)
            if overlap_bbox is None:
                continue

            rel_overlap_bbox = cls._bbox_to_relative_bbox(overlap_bbox, table_bbox)
            score = float(layout_item.get("score", 1.0))

            if label == "image":
                image_src = cls._encode_table_inline_image(np_img, item_bbox)
                if not image_src:
                    continue
                content = f'<img src="{image_src}"/>'
                token_bbox = cls._get_virtual_image_bbox(rel_overlap_bbox)
                kind = "image"
            else:
                latex = layout_item.get("latex", "")
                if not latex:
                    continue
                content = f"<eq>{html.escape(latex)}</eq>"
                token_bbox = rel_overlap_bbox
                kind = "formula"

            table_inline_objects[id(table_res)].append(
                {
                    "kind": kind,
                    "page_bbox": item_bbox,
                    "table_rel_mask_bbox": rel_overlap_bbox,
                    "table_token_bbox": token_bbox,
                    "content": content,
                    "score": score,
                }
            )
            remove_ids.add(id(layout_item))

        if remove_ids:
            layout_res[:] = [item for item in layout_res if id(item) not in remove_ids]

        return table_inline_objects


    # 真正的本地模型调用入口！！！！
    def __call__(self, images_with_extra_info: list) -> list:
        # 输入 images_with_extra_info 是 [(PIL_Image, ocr_enable, lang), ...] 的列表
        if len(images_with_extra_info) == 0:
            return []

        images_layout_res = []

        self.model = self.model_manager.get_model( # 加载组合模型(layout+mfr)，根据 formula/table 开关加载不同的模型组合
            lang=None,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
        )
        atom_model_manager = AtomModelSingleton() # 获取原子模型单例管理器，按需懒加载各子模型

        pil_images = [image for image, _, _ in images_with_extra_info] # 提取 PIL 图片列表

        np_images = [np.asarray(image) for image, _, _ in images_with_extra_info] # 转换为 NumPy 数组列表

        # 阶段1. 版面分析
        # 使用 pp-doclayout_v2 模型对所有页面图片做版面检测，
        # 输出每页一个 layout_res 列表，每个元素是 {"label": "text"|"table"|"image"|"display_formula"|..., "bbox": [...], "score": ...}
        images_layout_res += self.model.layout_model.batch_predict(
            pil_images,
            batch_size=min(8, self.batch_ratio * LAYOUT_BASE_BATCH_SIZE) # batch_size 上限卡 8，防止显存溢出
        )
        # 清理显存
        clean_vram(self.model.device, vram_threshold=8)

        # 阶段2. 公式识别
        if self.formula_enable:
            images_mfd_res = []
            for layout_res in images_layout_res:  # 遍历每页 layout，收集所有公式元素，初始化 latex=""
                page_formula_res = []
                for res in layout_res:
                    if res.get("label") in ["display_formula", "inline_formula"]:
                        res.setdefault("latex", "")
                        page_formula_res.append(res)
                images_mfd_res.append(page_formula_res) # 从 layout 结果中收集所有 display_formula 和 inline_formula

            # 批量送入 MFR 模型，得到 LaTeX
            images_formula_list = self.model.mfr_model.batch_predict( 
                images_mfd_res,
                np_images,
                batch_size=self.batch_ratio * MFR_BASE_BATCH_SIZE,
            )
            mfr_count = 0
            for image_index in range(len(np_images)):
                mfr_count += len(images_formula_list[image_index])
                for formula_res, formula_with_latex in zip(
                    images_mfd_res[image_index], images_formula_list[image_index]
                ):
                    formula_res["latex"] = formula_with_latex.get("latex", "") # 将识别出的 LaTeX 回填到 layout_res 中对应公式元素的 latex 字段

            # 清理显存
            clean_vram(self.model.device, vram_threshold=8)

        else:
            for layout_res in images_layout_res:
                # 直接从 layout 中移除所有 inline_formula（保留 display_formula 作为占位）
                layout_res[:] = [res for res in layout_res if res.get("label") != "inline_formula"]


        # 阶段3. 数据分拣
        # 初始化两个全局收集列表，逐页处理
        ocr_res_list_all_page = [] # 文字区域的 OCR 任务队列（按页）
        table_res_list_all_page = [] # 表格区域的识别任务队列（跨页扁平化）
        for index in range(len(np_images)): # 遍历所有页面
            _, ocr_enable, _lang = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            np_img = np_images[index]
            #调用 _extract_table_inline_objects，
            # 将落在表格 bbox 内的 image/formula 元素从 layout_res 中移除，编码为 HTML token（<img> / <eq>），按 table_id 归类
            table_inline_objects = (
                self._extract_table_inline_objects( # 提取表格内嵌对象
                    layout_res,
                    np_img,
                    formula_enable=self.formula_enable,
                )
                if self.table_enable
                else {}
            )
            # 将 layout_res 拆分为 ocr_res_list（文本区域）和 table_res_list（表格区域）
            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res)
            )
            # 将 layout 元素按类型拆分：OCR 文本区、表格区、公式检测框
            ocr_res_list_all_page.append({'ocr_res_list':ocr_res_list,
                                          'lang':_lang,
                                          'ocr_enable':ocr_enable,
                                          'np_img':np_img,
                                          'single_page_mfdetrec_res':single_page_mfdetrec_res,
                                          'layout_res':layout_res,
                                          })

            for table_res in table_res_list:
                def get_crop_table_img(scale):
                    bbox = normalize_to_int_bbox(
                        [float(v) / float(scale) for v in table_res["bbox"]]
                    )
                    if bbox is None:
                        return np_img[0:0, 0:0]
                    return get_crop_np_img(bbox, np_img, scale=scale)
                # 为每个表格裁剪两种分辨率的图片
                wireless_table_img = get_crop_table_img(scale = 1) # 原始分辨率
                wired_table_img = get_crop_table_img(scale = 10/3) # 3.33x 高分辨率
                table_page_bbox = normalize_to_int_bbox( # 表格区域
                    table_res.get("bbox"),
                    image_size=np_img.shape[:2],
                ) or [0, 0, 0, 0]

                table_res_list_all_page.append({'table_res':table_res,
                                                'lang':_lang,
                                                'table_img':wireless_table_img, # 用于无线表格识别
                                                'wired_table_img':wired_table_img, # 用于有线表格识别（高分辨率看线条）
                                                'table_page_bbox':table_page_bbox,
                                                'table_inline_objects':table_inline_objects.get(id(table_res), []),
                                              })

        # 阶段4. 表格识别流水线
        if self.table_enable:

            # 图片旋转批量处理
            # 4.1 判断表格图片是否旋转（0°/90°/180°/270°）并校正
            img_orientation_cls_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.ImgOrientationCls,
            )
            try:
                # 支持批处理和逐张两种模式
                if self.table_ori_cls_batch_enabled:
                    img_orientation_cls_model.batch_predict(table_res_list_all_page,
                                                            det_batch_size=self.batch_ratio * OCR_DET_BASE_BATCH_SIZE,
                                                            batch_size=TABLE_ORI_CLS_BATCH_SIZE)
                else:
                    for table_res in table_res_list_all_page:
                        rotate_label = img_orientation_cls_model.predict(table_res['table_img'])
                        img_orientation_cls_model.img_rotate(table_res, rotate_label)
            except Exception as e:
                logger.warning(
                    f"Image orientation classification failed: {e}, using original image"
                )

            # 表格分类
            # 4.2 表格有线/无线分类
            # 分类结果写入 table_res["cls_label"]（WiredTable / WirelessTable）和 table_res["cls_score"]
            table_cls_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.TableCls,
            )
            try:
                table_cls_model.batch_predict(table_res_list_all_page,
                                              batch_size=TABLE_Wired_Wireless_CLS_BATCH_SIZE)
            except Exception as e:
                logger.warning(
                    f"Table classification failed: {e}, using default model"
                )

            # OCR det 过程，顺序执行
            # 4.3 表格内 OCR 检测
            # 获取专用于表格的 OCR det 引擎（高阈值 0.5，不合并框）
            rec_img_lang_group = defaultdict(list)
            det_ocr_engine = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.OCR,
                det_db_box_thresh=0.5, # 较高阈值，表格文字一般比较清晰
                det_db_unclip_ratio=1.6,
                enable_merge_det_boxes=False, # 不合并检测框
            )
            for index, table_res_dict in enumerate( # 逐个表格处理
                    tqdm(table_res_list_all_page, desc="Table-ocr det")
            ):
                bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
                # 构建两组 mask：inline_mask_boxes（所有内嵌对象，用于遮盖后做 det）、formula_mask_boxes（仅公式，用于后续恢复检测框
                table_inline_objects = (
                    table_res_dict.get("table_inline_objects", [])
                    if self._table_supports_inline_objects(table_res_dict)
                    else []
                )
                inline_mask_boxes = [
                    {"bbox": inline_object["table_rel_mask_bbox"]}
                    for inline_object in table_inline_objects
                ]
                formula_mask_boxes = [
                    {"bbox": inline_object["table_rel_mask_bbox"]}
                    for inline_object in table_inline_objects
                    if inline_object["kind"] == "formula"
                ]
                # 先遮盖内嵌对象区域（涂白），再做文字检测
                det_image = (
                    self._apply_mask_boxes_to_image(bgr_image, inline_mask_boxes)
                    if inline_mask_boxes
                    else bgr_image
                )
                ocr_result = det_ocr_engine.ocr(det_image, rec=False)[0]
                # 如果有公式 mask，用 update_det_boxes 将公式区域的检测框补回来
                if ocr_result and formula_mask_boxes:
                    ocr_result = update_det_boxes(ocr_result, formula_mask_boxes)
                if ocr_result:
                    ocr_result = sorted_boxes(ocr_result)
                # 对每个检测框，从原始 BGR 图（未遮盖的）裁剪出文字行图片
                # 按语言分组，记录 table_id 用于后续回填
                for dt_box in ocr_result:
                    rec_img_lang_group[table_res_dict["lang"]].append(
                        {
                            "cropped_img": get_rotate_crop_image_for_text_rec(
                                bgr_image, np.asarray(dt_box, dtype=np.float32)
                            ),
                            "dt_box": np.asarray(dt_box, dtype=np.float32),
                            "table_id": index,
                        }
                    )

            # 4.4 表格内 OCR 识别
            # 按语言分别获取 OCR rec 模型，批量识别文字
            for _lang, rec_img_list in rec_img_lang_group.items():
                if not rec_img_list:
                    continue
                ocr_engine = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    det_db_box_thresh=0.5,
                    det_db_unclip_ratio=1.6,
                    lang=_lang,
                    enable_merge_det_boxes=False,
                )
                cropped_img_list = [item["cropped_img"] for item in rec_img_list]
                ocr_res_list = ocr_engine.ocr(cropped_img_list, det=False, tqdm_enable=True, tqdm_desc=f"Table-ocr rec {_lang}")[0]
                # 按 table_id 将 [检测框, HTML转义文字, 置信度] 回填到对应表格的 ocr_result
                for img_dict, ocr_res in zip(rec_img_list, ocr_res_list):
                    if table_res_list_all_page[img_dict["table_id"]].get("ocr_result"):
                        table_res_list_all_page[img_dict["table_id"]]["ocr_result"].append(
                            [img_dict["dt_box"], html.escape(ocr_res[0]), ocr_res[1]]
                        )
                    else:
                        table_res_list_all_page[img_dict["table_id"]]["ocr_result"] = [
                            [img_dict["dt_box"], html.escape(ocr_res[0]), ocr_res[1]]
                        ]

            # 先对所有表格使用无线表格模型，然后对分类为有线的表格使用有线表格模型
            # 4.5 将之前提取的内嵌图片/公式 HTML token 插入表格 OCR 结果
            for table_res_dict in table_res_list_all_page:
                if not self._table_supports_inline_objects(table_res_dict):
                    continue

                table_inline_objects = table_res_dict.get("table_inline_objects", [])
                if not table_inline_objects:
                    continue

                table_ocr_result = table_res_dict.setdefault("ocr_result", [])
                for inline_object in table_inline_objects:
                    table_ocr_result.append(
                        [
                            self._bbox_to_quad(inline_object["table_token_bbox"]),
                            inline_object["content"],
                            inline_object["score"],
                        ]
                    )

                self._sort_table_ocr_result(table_ocr_result) # 重新排序保证阅读顺序
            # 4.6 用无线表格模型对所有表格做批量预测，生成 HTML 结构
            wireless_table_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.WirelessTable,
            )
            wireless_table_model.batch_predict(table_res_list_all_page) # 结果写入 table_res["html"]

            # 4.7 单独拿出有线表格进行预测
            wired_table_res_list = []
            for table_res_dict in table_res_list_all_page:
                # 筛选条件（二选一）：
                # 1. 分类为 WirelessTable 但置信度 < 0.9（不太确定是无线的，兜底走有线再预测一遍）
                # 2. 分类为 WiredTable（明确是有线表格）
                # logger.debug(f"Table classification result: {table_res_dict["table_res"]["cls_label"]} with confidence {table_res_dict["table_res"]["cls_score"]}")
                if (
                    (table_res_dict["table_res"]["cls_label"] == AtomicModel.WirelessTable and table_res_dict["table_res"]["cls_score"] < 0.9)
                    or table_res_dict["table_res"]["cls_label"] == AtomicModel.WiredTable
                ):
                    wired_table_res_list.append(table_res_dict) # 满足条件的加入 wired_table_res_list
                del table_res_dict["table_res"]["cls_label"]
                del table_res_dict["table_res"]["cls_score"] # 清理临时分类标签
            if wired_table_res_list: # 仅当存在有线表格时才执行
                for table_res_dict in tqdm(
                        wired_table_res_list, desc="Table-wired Predict"
                ):
                    # 跳过没有 OCR 结果的表格（步骤 4.3-4.4 没检测到任何文字的表格，有线模型也无法处理）
                    if not table_res_dict.get("ocr_result", None):
                        continue
                    # 按语言获取有线表格模型（不同语言可能有不同的模型权重）    
                    wired_table_model = atom_model_manager.get_atom_model(
                        atom_model_name=AtomicModel.WiredTable,
                        lang=table_res_dict["lang"],
                    )
                    # 三个输入：
                    # wired_table_img：10/3x 高分辨率裁剪图（步骤 3c L402 裁的，放大后线条更清晰）
                    # ocr_result：步骤 4.3-4.5 产出的 [quad, text, score] 列表（含内嵌公式/图片 HTML）
                    # html：步骤 4.6 无线模型已生成的 HTML 结构（作为参考/fallback）  
                    table_res_dict["table_res"]["html"] = wired_table_model.predict(
                        table_res_dict["wired_table_img"],
                        table_res_dict["ocr_result"],
                        table_res_dict["table_res"].get("html", None)
                    )

            # 4.8 表格格式清理
            for table_res_dict in table_res_list_all_page:
                html_code = table_res_dict["table_res"].get("html", "") or "" # 遍历所有表格，取出 HTML

                # 检查html_code是否包含'<table>'和'</table>'
                # 用 find / rfind 定位第一个 <table> 和最后一个 </table>
                # 只保留这个区间内的内容，去掉模型可能生成的前缀说明文字或尾部垃圾
                if "<table>" in html_code and "</table>" in html_code:
                    # 选用<table>到</table>的内容，放入table_res_dict['table_res']['html']
                    start_index = html_code.find("<table>")
                    end_index = html_code.rfind("</table>") + len("</table>")
                    table_res_dict["table_res"]["html"] = html_code[start_index:end_index]


        # 步骤5. 正文文字 OCR 检测
        if self.text_ocr_det_batch_enabled: # 根据配置选择批处理模式或逐张模式
            # 批处理模式 - 按语言和分辨率分组
            # 5.1 收集所有需要OCR检测的裁剪图像
            all_cropped_images_info = []

            for ocr_res_list_dict in ocr_res_list_all_page: # 遍历每个页面的每个待 OCR 文本区域（步骤 3b 中拆出的 ocr_res_list）
                _lang = ocr_res_list_dict['lang']

                for res in ocr_res_list_dict['ocr_res_list']:
                    # crop_img：根据 res["bbox"] 从原图裁剪，四周各扩展 50px（防止文字被截断）
                    # useful_list：记录裁剪坐标映射关系，后续将检测框坐标映射回原图坐标用
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    # 将该页的公式检测框 (single_page_mfdetrec_res) 也做同样的坐标变换，映射到裁剪图的局部坐标系
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                    )

                    # BGR转换
                    # RGB→BGR 转换（OpenCV 格式）
                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                    # 如果 mask_inline_formula_for_ocr_det=True，则调用 _apply_mask_boxes_to_image 将公式区域涂白；否则返回原图
                    det_image = self._get_masked_det_image(
                        bgr_image,
                        adjusted_mfdetrec_res,
                    )

                    all_cropped_images_info.append((
                        bgr_image,
                        det_image,
                        useful_list,
                        ocr_res_list_dict,
                        adjusted_mfdetrec_res,
                        _lang,
                    ))

            # 5.2 按语言分组
            lang_groups = defaultdict(list)
            for crop_info in all_cropped_images_info:
                lang = crop_info[5]
                lang_groups[lang].append(crop_info)

            # 对每种语言按分辨率分组并批处理
            for lang, lang_crop_list in lang_groups.items():
                if not lang_crop_list:
                    continue

                # logger.info(f"Processing OCR detection for language {lang} with {len(lang_crop_list)} images")

                # 获取OCR模型
                ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    det_db_box_thresh=0.3,
                    lang=lang
                )

                # 按分辨率分组并同时完成padding
                # RESOLUTION_GROUP_STRIDE = 32
                # 将高度和宽度各自向上对齐到 64 的倍数，对齐后尺寸相同的图片归为一组
                # 目的：同组图片 pad 到相同尺寸后可以组成 batch（GPU 批处理要求张量维度一致）
                # STRIDE 选 64（曾尝试 32，注释还留着），64 意味着分组更粗，同组图片更多，batch 更大，但 padding 浪费也稍多
                RESOLUTION_GROUP_STRIDE = 64

                resolution_groups = defaultdict(list)
                for crop_info in lang_crop_list:
                    cropped_img = crop_info[1]
                    h, w = cropped_img.shape[:2]
                    # 直接计算目标尺寸并用作分组键
                    target_h = ((h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                    target_w = ((w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                    group_key = (target_h, target_w)
                    resolution_groups[group_key].append(crop_info)

                # 对每个分辨率组进行批处理
                for (target_h, target_w), group_crops in tqdm(resolution_groups.items(), desc=f"OCR-det {lang}"):
                    # 对所有图像进行padding到统一尺寸
                    batch_images = []
                    for crop_info in group_crops:
                        img = crop_info[1]
                        h, w = img.shape[:2]
                        # 创建目标尺寸的白色背景
                        padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                        padded_img[:h, :w] = img
                        batch_images.append(padded_img)

                    # 批处理检测
                    det_batch_size = min(len(batch_images), self.batch_ratio * OCR_DET_BASE_BATCH_SIZE)
                    batch_results = ocr_model.text_detector.batch_predict(batch_images, det_batch_size)

                    # 处理批处理结果
                    for crop_info, (dt_boxes, _) in zip(group_crops, batch_results):
                        (
                            bgr_image,
                            _det_image,
                            useful_list,
                            ocr_res_list_dict,
                            adjusted_mfdetrec_res,
                            _lang,
                        ) = crop_info

                        if dt_boxes is not None and len(dt_boxes) > 0:
                            # 处理检测框
                            dt_boxes_sorted = sorted_boxes(dt_boxes)
                            # 合并相邻的检测框（将紧挨着的碎框合成一个大框，减少后续 rec 的调用次数）
                            dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []

                            # 根据公式位置更新检测框
                            dt_boxes_final = (update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                                              if dt_boxes_merged and adjusted_mfdetrec_res
                                              else dt_boxes_merged)

                            # 用 useful_list 将裁剪图坐标映射回原图坐标，为每个检测框从 bgr_image 中裁剪出文字行小图，生成 layout 元素 {"label": "ocr_text", "bbox": [...], "_need_ocr_rec": True, "np_img": 裁剪图, "lang": ...}
                            if dt_boxes_final:
                                ocr_res = [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes_final]
                                ocr_result_list = get_ocr_result_list(
                                    ocr_res,
                                    useful_list,
                                    ocr_res_list_dict['ocr_enable'],
                                    bgr_image,
                                    _lang,
                                )
                                ocr_res_list_dict['layout_res'].extend(ocr_result_list)

            # 清理显存
            clean_vram(self.model.device, vram_threshold=8)

        else:
            # 原始单张处理模式，每页获取对应语言的 OCR 模型
            for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="OCR-det Predict"):
                # Process each area that requires OCR processing
                _lang = ocr_res_list_dict['lang']
                # Get OCR results for this language's images
                ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    ocr_show_log=False,
                    det_db_box_thresh=0.3,
                    lang=_lang
                )
                for res in ocr_res_list_dict['ocr_res_list']:
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                    )
                    # OCR-det， 裁剪 + 调整公式框坐标
                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                    det_image = self._get_masked_det_image(
                        bgr_image,
                        adjusted_mfdetrec_res,
                    )
                    ocr_res = ocr_model.ocr(
                        det_image, mfd_res=adjusted_mfdetrec_res, rec=False
                    )[0]
                    # 与批处理模式的区别：
                    # 批处理模式调 text_detector.batch_predict，自己处理 sorted/merge/update
                    # 逐张模式调 ocr(rec=False)，传入 mfd_res，ocr 内部做 sorted/merge/update（黑盒）

                    # Integration results
                    if ocr_res:
                        ocr_result_list = get_ocr_result_list(
                            ocr_res,
                            useful_list,
                            ocr_res_list_dict['ocr_enable'],
                            bgr_image,
                            _lang,
                        )

                        ocr_res_list_dict['layout_res'].extend(ocr_result_list)

        # 6. 正文文字 OCR 识别
        # Create dictionaries to store items by language
        need_ocr_lists_by_lang = {}  # Dict of lists for each language
        img_crop_lists_by_lang = {}  # Dict of lists for each language
        # 6.1 收集待识别元素，按语言分组
        for layout_res in images_layout_res:
            for layout_res_item in layout_res:
                if not layout_res_item.get("_need_ocr_rec"): # get_ocr_result_list 标记了 _need_ocr_rec=True 的元素
                    continue
                if 'np_img' in layout_res_item and 'lang' in layout_res_item:
                    lang = layout_res_item['lang']

                    # Initialize lists for this language if not exist
                    if lang not in need_ocr_lists_by_lang:
                        need_ocr_lists_by_lang[lang] = []
                        img_crop_lists_by_lang[lang] = []

                    # Add to the appropriate language-specific lists
                    need_ocr_lists_by_lang[lang].append((layout_res, layout_res_item)) # 保留对 (page_layout_res, item) 的引用，方便后续回填和删除
                    img_crop_lists_by_lang[lang].append(layout_res_item['np_img']) # 裁剪图列表，直接送入 OCR rec

                    # Remove temporary fields after collecting 立即清理临时字段，减少内存占用
                    layout_res_item.pop('np_img', None)
                    layout_res_item.pop('lang', None)
                    layout_res_item.pop('_need_ocr_rec', None)

        # 6.2 按语言分别获取 OCR 模型，批量识别文字
        if len(img_crop_lists_by_lang) > 0:

            # Process OCR by language
            total_processed = 0

            # Process each language separately
            for lang, img_crop_list in img_crop_lists_by_lang.items():
                if len(img_crop_list) > 0:
                    # Get OCR results for this language's images

                    ocr_model = atom_model_manager.get_atom_model(
                        atom_model_name=AtomicModel.OCR,
                        det_db_box_thresh=0.3,
                        lang=lang
                    )
                    ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]

                    # Verify we have matching counts
                    # 安全断言：输出数量必须等于输入数量
                    assert len(ocr_res_list) == len(
                        need_ocr_lists_by_lang[lang]), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_lists_by_lang[lang])} for lang: {lang}'

                    items_to_remove = []
                    # 回填结果 + 质量过滤
                    # Process OCR results for this language
                    for index, (page_layout_res, layout_res_item) in enumerate(need_ocr_lists_by_lang[lang]):
                        ocr_text, ocr_score = ocr_res_list[index]
                        layout_res_item['text'] = ocr_text
                        layout_res_item['score'] = float(f"{ocr_score:.3f}") # 逐个回填文字和置信度
                        should_remove = False
                        if ocr_score < OcrConfidence.min_confidence:
                            should_remove = True
                        else: # 过滤规则 1：置信度低于全局最低阈值（OcrConfidence.min_confidence）→ 标记移除
                            layout_res_bbox = layout_res_item['bbox']
                            layout_res_width = layout_res_bbox[2] - layout_res_bbox[0]
                            layout_res_height = layout_res_bbox[3] - layout_res_bbox[1]
                            # 过滤规则 2：三个条件同时满足时才移除：
                            # 1. 文字命中硬编码噪声列表——这些是 PDF 嵌入字体 CID 映射错误的常见产物
                            # 2. 置信度 < 0.8（不太可信）
                            # 3. 区域宽度 < 高度（竖条状，通常是页边装饰/页码区域的噪声）
                            if (
                                    ocr_text in [
                                        '（204号', '（20', '（2', '（2号', '（20号', '号', '（204',
                                        '(cid:)', '(ci:)', '(cd:1)', 'cd:)', 'c)', '(cd:)', 'c', 'id:)',
                                        ':)', '√:)', '√i:)', '−i:)', '−:', 'i:)',
                                    ]
                                    and ocr_score < 0.8
                                    and layout_res_width < layout_res_height
                            ):
                                should_remove = True

                        if should_remove:
                            items_to_remove.append((page_layout_res, layout_res_item))

                    for page_layout_res, layout_res_item in items_to_remove:
                        if layout_res_item in page_layout_res:
                            page_layout_res.remove(layout_res_item) # 批量从各页的 layout_res 中移除不合格元素

                    total_processed += len(img_crop_list)

        # 步骤7. 印章文字 OCR 检测
        seal_ocr_items = []
        # 7.1 收集待检测的印章区域
        for ocr_res_list_dict in ocr_res_list_all_page:
            for layout_res_item in ocr_res_list_dict['layout_res']:
                if layout_res_item.get("label") == "seal": # 只处理 label="seal" 的印章区域
                    seal_ocr_items.append((ocr_res_list_dict, layout_res_item))

        seal_ocr_model = None
        for ocr_res_list_dict, layout_res_item in tqdm(seal_ocr_items, desc="Seal Predict"):
            np_img = ocr_res_list_dict['np_img']
            image_h, image_w = np_img.shape[:2]
            layout_res_item["text"] = "" # 初始化 text 字段为空字符串，后续回填识别结果
            seal_bbox = normalize_to_int_bbox(
                layout_res_item.get("bbox"),
                image_size=(image_h, image_w),
            )
            if seal_bbox is None:
                continue

            x0, y0, x1, y1 = seal_bbox # bbox 归一化为整数坐标，无效则跳过
            seal_crop_rgb = np_img[y0:y1, x0:x1]
            if seal_crop_rgb.size == 0:
                continue

            if seal_ocr_model is None: # 裁剪印章区域，跳过空裁剪
                seal_ocr_model = atom_model_manager.get_atom_model( # 首次使用时懒加载 lang="seal" 的专用 OCR 模型，印章文字通常是弯曲环形排列，需要特殊的检测和识别处理
                    atom_model_name=AtomicModel.OCR,
                    lang="seal",
                )
            # 对印章裁剪图同时做检测 + 识别
            seal_crop_bgr = cv2.cvtColor(seal_crop_rgb, cv2.COLOR_RGB2BGR)
            seal_ocr_res = seal_ocr_model.ocr(seal_crop_bgr, det=True, rec=True)[0]
            if not seal_ocr_res:
                continue

            # 提取所有识别出的文字行，作为 list 存入 layout_res_item["text"]
            # 注意：印章的 text 是 list[str]，而普通文本的 text 是 str
            seal_texts = []
            for seal_item in seal_ocr_res:
                if not seal_item or len(seal_item) != 2:
                    continue
                rec_result = seal_item[1]
                if not rec_result or len(rec_result) < 1:
                    continue
                rec_text = rec_result[0]
                if rec_text:
                    seal_texts.append(rec_text)

            layout_res_item["text"] = seal_texts

        # 步骤8. 空文本块清理
        for ocr_res_list_dict in ocr_res_list_all_page: # 移除 label="ocr_text" 但 text 为空或仅空白的元素
            self._prune_empty_ocr_text_blocks(
                ocr_res_list_dict["layout_res"], # 仅在 ocr_enable=True 时执行（否则没有 ocr_text 类型的元素）
                ocr_res_list_dict["ocr_enable"],
            )

        return images_layout_res
