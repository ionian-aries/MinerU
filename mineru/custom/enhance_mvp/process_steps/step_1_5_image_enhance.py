"""Stage 1.5: ImageEnhanceStage — 多模态 LLM 为图片 BlockUnit 生成语义描述。

插入位置：Stage 1（build_block_items）之后、Stage 2（SectionPlan）之前。
必须在 Stage 2 之前完成：Stage 2 的 token 预算计算依赖 BodyItem.char_count，
而 char_count 由 BlockUnit.char_count 聚合，增强描述追加后需立即生效。

设计原则：
- 仅处理 bu.img_path 非空的 BlockUnit（image/chart，或 table without HTML）。
- TABLE with HTML / CODE / 普通块：Stage 1 已将 img_path="" → `if not bu.img_path` 自动跳过。
- 失败是真正的 no-op：bu.rendered_text 不变，对 Stage 2 透明。
- 无 text-only 降级模式：provider 不支持 vision → 整个 Stage 1.5 跳过。
- 并发调用（ThreadPoolExecutor），每张图 2 次尝试，失败不阻断 pipeline。
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .step_1_build_block_items import BlockUnit, BodyItem, TitleItem

# ── 质量检查常量 ────────────────────────────────────────────────────────────
_MIN_DESC_LEN = 20   # 增强描述最短字符数，低于此视为质量不合格

# ── System Prompt ───────────────────────────────────────────────────────────
_IMAGE_ENHANCE_SYSTEM_PROMPT = (
    "你是技术文档图片分析助手。"
    "请先判断图片类型（数据图表/流程图/架构图/表格截图/实物照片/公式/其他），"
    "再依据类型选择描述侧重：\n"
    "- 数据图表/折线图/柱状图：趋势、轴标签、关键数值\n"
    "- 流程图/架构图/拓扑图：组件、层次关系、流向\n"
    "- 表格截图：表头含义、关键数据点\n"
    "- 实物照片/产品图：对象特征、状态、标注信息\n"
    "- 公式/方程：完整转录公式内容\n"
    "描述须自包含（禁止'如图所示''本图'等指代），保留所有数值、单位、编号。"
)


@dataclass
class _ImageContext:
    """单张图的增强上下文，供 LLM 调用时构建 prompt。"""
    item_index: int
    bu_index: int
    img_path: str
    caption_text: str
    footnote_text: str
    section_title: str
    text_before: str   # 图前上下文（≤250字，含其他图片占位符）
    text_after: str    # 图后上下文（≤250字，含其他图片占位符）


def _tail_chars(texts: list[str], max_chars: int) -> str:
    """从文本列表末尾收集不超过 max_chars 个字符（保持原顺序）。"""
    if not texts:
        return ""
    collected: list[str] = []
    remaining = max_chars
    for text in reversed(texts):
        if remaining <= 0:
            break
        chunk = text[-remaining:] if len(text) > remaining else text
        collected.append(chunk)
        remaining -= len(chunk)
    return "".join(reversed(collected))


def _head_chars(texts: list[str], max_chars: int) -> str:
    """从文本列表头部收集不超过 max_chars 个字符。"""
    if not texts:
        return ""
    collected: list[str] = []
    remaining = max_chars
    for text in texts:
        if remaining <= 0:
            break
        chunk = text[:remaining]
        collected.append(chunk)
        remaining -= len(chunk)
    return "".join(collected)


def _min_dimension_from_refs(block_refs: list[list]) -> float:
    """从 block_refs 中解析 bbox，返回所有子块 bbox 最短边的最小值。"""
    min_side = float("inf")
    for ref in block_refs:
        if not isinstance(ref, list) or len(ref) < 4:
            continue
        bbox_str = ref[3] if isinstance(ref[3], str) else ""
        if not bbox_str or "," not in bbox_str:
            continue
        try:
            parts = [float(v) for v in bbox_str.split(",")[:4]]
            w = abs(parts[2] - parts[0])
            h = abs(parts[3] - parts[1])
            min_side = min(min_side, w, h)
        except (ValueError, IndexError):
            continue
    return min_side if min_side != float("inf") else 0.0


def _build_user_prompt(ctx: _ImageContext) -> str:
    """构建用于图片增强的 user prompt。"""
    parts: list[str] = []
    if ctx.section_title:
        parts.append(f"章节标题：{ctx.section_title}")
    if ctx.caption_text:
        parts.append(f"图题：{ctx.caption_text}")
    if ctx.footnote_text:
        parts.append(f"图注：{ctx.footnote_text}")
    if ctx.text_before.strip():
        parts.append(f"前文：{ctx.text_before.strip()}")
    if ctx.text_after.strip():
        parts.append(f"后文：{ctx.text_after.strip()}")
    parts.append("请生成该图片的语义描述（自包含，技术密集，不使用'如图所示'等指代）：")
    return "\n".join(parts)


def _enhance_single_image(
    *,
    ctx: _ImageContext,
    provider: Any,
    max_tokens: int,
) -> str | None:
    """对单张图调用多模态 LLM，返回增强描述文字，失败返回 None。

    尝试 2 次，均失败则返回 None（no-op）。
    """
    user_prompt = _build_user_prompt(ctx)

    for attempt in range(2):
        try:
            result = provider.generate_with_image(
                prompt=user_prompt,
                image_path=ctx.img_path,
                system_prompt=_IMAGE_ENHANCE_SYSTEM_PROMPT,
                max_tokens=max_tokens,
            )
            text = str(result).strip() if result else ""
            if len(text) >= _MIN_DESC_LEN:
                return text
            logger.warning(
                f"[ImageEnhance] quality check failed (len={len(text)}) "
                f"item={ctx.item_index} bu={ctx.bu_index} attempt={attempt + 1}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[ImageEnhance] LLM call failed item={ctx.item_index} "
                f"bu={ctx.bu_index} attempt={attempt + 1}: {exc}"
            )
    return None


def run_image_enhance_stage(
    *,
    items: list[TitleItem | BodyItem],
    provider: Any,
    image_enhance_cfg: dict[str, Any],
    image_dir: str = "",
) -> tuple[list[TitleItem | BodyItem], dict[str, dict]]:
    """Stage 1.5：ImageEnhanceStage。

    - 遍历 items，为每个 img_path 非空的 image/table-as-image BlockUnit 并发调用多模态 LLM。
    - 成功：bu.rendered_text += "\\n" + description；bu.char_count / bu.image_description 同步更新。
    - 失败：no-op，bu.rendered_text 不变。
    - 返回 (items, image_enhance_records)：
        items：原列表（in-place 修改，引用不变）
        image_enhance_records：{img_filename: {context, params, description, status}}
    """
    concurrency: int = max(1, int(image_enhance_cfg.get("concurrency", 2) or 2))
    max_tokens: int = max(50, int(image_enhance_cfg.get("max_tokens", 300) or 300))
    min_short_side: float = float(image_enhance_cfg.get("min_short_side_px", 10) or 10)

    # ── Phase 0: vision support check ────────────────────────────────────────
    if not callable(getattr(provider, "generate_with_image", None)):
        logger.info(
            "[ImageEnhance] provider does not support generate_with_image, "
            "Stage 1.5 skipped entirely."
        )
        return items

    # ── Phase 1: 滚动扫描 items，建立 section_title_index ───────────────────
    section_title_index: dict[int, str] = {}
    current_title = ""
    for i, item in enumerate(items):
        if isinstance(item, TitleItem):
            current_title = item.title_text
        else:
            section_title_index[i] = current_title

    # ── Phase 2: 逐 BodyItem → 逐 BlockUnit，收集待增强任务 ─────────────────
    all_tasks: list[tuple[int, int, _ImageContext]] = []
    _stat_total_visual_bu = 0     # 所有需要考察的视觉块（IMAGE/CHART + TABLE-as-image）总数
    _stat_skip_no_path = 0        # img_path 为空（TABLE with HTML / CODE）跳过数
    _stat_skip_tiny = 0           # 尺寸过小跳过数

    for i, item in enumerate(items):
        if not isinstance(item, BodyItem):
            continue
        if item.image_count == 0 and item.table_count == 0:
            continue  # 快速跳过无图无表 BodyItem

        section_title = section_title_index.get(i, "")

        for j, bu in enumerate(item.block_items):
            # IMAGE/CHART → image_count=1
            # TABLE without HTML (table_body 仅有 image_path，无 html) → table_count=1 且 img_path 非空
            is_visual = (bu.image_count == 1) or (bu.table_count == 1 and bool(bu.img_path))
            if not is_visual:
                continue
            _stat_total_visual_bu += 1
            if not bu.img_path:
                _stat_skip_no_path += 1
                continue  # TABLE with HTML / CODE / 普通块（img_path=""）

            # 防御性尺寸检查
            short_side = _min_dimension_from_refs(bu.block_refs)
            if short_side > 0 and short_side < min_short_side:
                logger.debug(
                    f"[ImageEnhance] skip tiny block item={i} bu={j} "
                    f"short_side={short_side:.1f} < {min_short_side}"
                )
                _stat_skip_tiny += 1
                continue

            # 从相邻 BlockUnit 提取上下文
            before_texts = [b.rendered_text for b in item.block_items[:j] if b.rendered_text]
            after_texts = [b.rendered_text for b in item.block_items[j + 1:] if b.rendered_text]

            ctx = _ImageContext(
                item_index=i,
                bu_index=j,
                img_path=(
                    os.path.join(image_dir, bu.img_path)
                    if image_dir and bu.img_path
                    else bu.img_path
                ),
                caption_text=bu.caption_text,
                footnote_text=bu.footnote_text,
                section_title=section_title,
                text_before=_tail_chars(before_texts, 250),
                text_after=_head_chars(after_texts, 250),
            )
            all_tasks.append((i, j, ctx))

    if not all_tasks:
        logger.info(
            f"[ImageEnhance] no image tasks, Stage 1.5 done (no-op). "
            f"visual_bu={_stat_total_visual_bu} skip_no_path={_stat_skip_no_path} "
            f"skip_tiny={_stat_skip_tiny}"
        )
        return items, {}

    logger.info(f"[ImageEnhance] start: tasks={len(all_tasks)} concurrency={concurrency}")

    # ── Phase 3: 并发 LLM 调用 ───────────────────────────────────────────────
    results: dict[tuple[int, int], str | None] = {}

    if concurrency <= 1 or len(all_tasks) == 1:
        for i, j, ctx in all_tasks:
            results[(i, j)] = _enhance_single_image(
                ctx=ctx, provider=provider, max_tokens=max_tokens
            )
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {
                executor.submit(
                    _enhance_single_image, ctx=ctx, provider=provider, max_tokens=max_tokens
                ): (i, j)
                for i, j, ctx in all_tasks
            }
            for future in as_completed(future_map):
                key = future_map[future]
                try:
                    results[key] = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[ImageEnhance] future failed {key}: {exc}")
                    results[key] = None

    # ── Phase 4: 回写 BlockUnit，同时构建 image_enhance_records ─────────────
    success_count = 0
    fail_count = 0
    task_map: dict[tuple[int, int], _ImageContext] = {
        (i, j): ctx for i, j, ctx in all_tasks
    }
    image_enhance_records: dict[str, dict] = {}

    for (i, j), enhanced_text in results.items():
        ctx = task_map[(i, j)]
        bu: BlockUnit = items[i].block_items[j]  # type: ignore[index]

        status = "success" if enhanced_text else "failed"
        # img_path 在 bu 上是裸文件名（uuid.ext），作为 enhance.json 的键
        image_enhance_records[bu.img_path] = {
            "context": {
                "section_title": ctx.section_title,
                "caption_text":  ctx.caption_text,
                "footnote_text": ctx.footnote_text,
                "text_before":   ctx.text_before,
                "text_after":    ctx.text_after,
            },
            "params": {
                "max_tokens": max_tokens,
            },
            "description": enhanced_text or None,
            "status": status,
        }

        if not enhanced_text:
            fail_count += 1
            continue  # no-op

        bu.rendered_text = bu.rendered_text + "\n" + enhanced_text
        bu.char_count = len(bu.rendered_text)
        bu.image_description = enhanced_text      # 供 ComposeStage 注入 enhanced.md
        # BodyItem.char_count / body_text 均为 @property → 自动聚合更新
        success_count += 1

    logger.info(
        f"[ImageEnhance] done: total={len(all_tasks)} "
        f"success={success_count} failed={fail_count}"
    )
    return items, image_enhance_records
