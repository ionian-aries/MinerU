"""Stage 6: ValidateStage — schema + 内容 + 引用三类校验。

对 section_results 和 doc_overview 做完整校验:
1. Schema 校验 (Pydantic)
2. 内容校验 (摘要非空, 关键词数量下限)
3. 引用校验 (id + refs 完整性)

失败策略: 标记 status=failed, 不中断流程。
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from mineru.custom.enhance_mvp.schema import validate_doc_payload, validate_section_payload


def _collect_section_ids_flat(section_results: list[dict[str, Any]] | None) -> set[str]:
    out: set[str] = set()

    def walk(nodes: list[dict[str, Any]] | None) -> None:
        for n in nodes or []:
            if not isinstance(n, dict):
                continue
            sid = n.get("id")
            if sid:
                out.add(str(sid))
            subs = n.get("subsections")
            if isinstance(subs, list):
                walk([s for s in subs if isinstance(s, dict)])

    walk(section_results)
    return out


def _sanitize_doc_outline_refs(doc: dict[str, Any], valid_ids: set[str]) -> dict[str, Any]:
    """丢弃 section_id 不在树中的洞察项，避免错误引用。"""
    if not valid_ids:
        return doc
    d = dict(doc)
    insights = d.get("doc_outline_insights")
    if not isinstance(insights, list):
        return doc
    fixed: list[dict[str, str]] = []
    for item in insights:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        sid = str(item.get("section_id", "")).strip()
        if not text or not sid:
            continue
        if sid not in valid_ids:
            logger.warning(
                f"DocEnhance: drop doc_outline_insights item with unknown section_id={sid!r}"
            )
            continue
        fixed.append({"text": text, "section_id": sid})
    out = dict(doc)
    out["doc_outline_insights"] = fixed
    return out


def _validate_section_item(item: dict[str, Any]) -> dict[str, Any]:
    """校验单个 section 结果, 失败则降级为 failed。"""
    if item.get("status") != "success":
        # 即使父节点失败，也尽量递归校验其 subsections（容错，不中断）。
        subsections = item.get("subsections")
        if isinstance(subsections, list):
            item = dict(item)
            item["subsections"] = [_validate_section_item(dict(s)) for s in subsections if isinstance(s, dict)]
        return item  # 已经 failed, 不再做主内容校验

    # 1. Schema + 内容校验
    payload_for_validation = {
        "summary": item.get("summary", ""),
        "keywords": item.get("keywords", []),
        "main_idea": item.get("main_idea", ""),
    }
    ok, reason = validate_section_payload(payload_for_validation)
    if not ok:
        section_id = item.get("id", "?")
        logger.warning(f"Section {section_id} schema/content validation failed: {reason}")
        item = dict(item)
        item["status"] = "failed"
        item["error"] = f"validate_schema: {reason}"
        return item

    # 2. 引用校验: 必须有 id 和 refs
    section_id = item.get("id", "")
    refs = item.get("refs")
    if not section_id:
        logger.warning("Section missing id")
        item = dict(item)
        item["status"] = "failed"
        item["error"] = "validate_ref: missing id"
        return item
    if not refs or not isinstance(refs, list) or len(refs) == 0:
        logger.warning(f"Section {section_id} has empty refs")
        item = dict(item)
        item["status"] = "failed"
        item["error"] = "validate_ref: empty refs"
        return item

    # 3. 关键词数量下限 (至少 1 个)
    keywords = item.get("keywords", [])
    if not keywords or len(keywords) < 1:
        logger.warning(f"Section {section_id} has no keywords")
        item = dict(item)
        item["status"] = "failed"
        item["error"] = "validate_content: no keywords"
        return item

    # 4. 递归校验 subsections
    subsections = item.get("subsections")
    if isinstance(subsections, list) and subsections:
        item = dict(item)
        item["subsections"] = [_validate_section_item(dict(s)) for s in subsections if isinstance(s, dict)]

    return item


def run_validate_stage(
    doc_overview: Any,
    section_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """完整校验入口。

    参数:
        doc_overview: DocEnhanceStage 的输出（global_summary/global_keywords/doc_outline_insights/answerable_questions）
        section_results: SectionEnhanceStage 的输出列表 (会被 **原地修改**)

    返回:
        校验后的 doc_overview (失败时返回降级兜底)
    """
    # ---- section 级校验 (原地修改 status) ----
    if section_results is not None:
        for i, item in enumerate(section_results):
            section_results[i] = _validate_section_item(item)

    # ---- doc 级校验 ----
    doc_dict = doc_overview if isinstance(doc_overview, dict) else {}
    valid_ids = _collect_section_ids_flat(section_results)
    if valid_ids:
        doc_dict = _sanitize_doc_outline_refs(doc_dict, valid_ids)

    ok_doc, doc_reason = validate_doc_payload(doc_dict)
    if not ok_doc:
        logger.warning(f"DocEnhance validation failed: {doc_reason}")
        return {
            "global_summary": "未生成",
            "global_keywords": [],
            "doc_outline_insights": [],
            "answerable_questions": [],
        }

    # 仅输出契约四键，不携带模型可能返回的其它字段
    return {
        "global_summary": doc_dict.get("global_summary", ""),
        "global_keywords": doc_dict.get("global_keywords", []),
        "doc_outline_insights": doc_dict.get("doc_outline_insights", []),
        "answerable_questions": doc_dict.get("answerable_questions", []),
    }
