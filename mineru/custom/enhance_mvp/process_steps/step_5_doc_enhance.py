"""Stage 5: DocEnhanceStage — 全文汇总增强。

使用 L1 节点的增强结果（section_results 顶层列表），
支持 Mode A（单次）/ Mode B（分组 map-reduce）自适应。

使用统一的 TokenCounter（来自 utils.py）。
"""
import json
import re
from typing import Any

from loguru import logger

from mineru.custom.enhance_mvp.prompts import render_prompt
from mineru.custom.enhance_mvp.utils import TokenCounter

# ---------------------------------------------------------------------------
# System prompt & retry 约束常量
# ---------------------------------------------------------------------------

_DOC_SYSTEM_PROMPT = "你是文档汇总助手。必须返回JSON对象。"

_RETRY_TITLES_DEGENERATED = (
    "\n\n额外约束：doc_outline_insights 中每条 text 必须是对主题的抽象概括，"
    "不要直接复述章节标题、目录项或原文标题；section_id 仍须对应输入中的 id。"
)

_RETRY_SUMMARIES_DEGENERATED = (
    "\n\n【严重问题】你的 doc_outline_insights 只是各章节 summary 的复述或抽句，"
    "这不符合要求。doc_outline_insights 必须是更高抽象层次的跨章节洞察：\n"
    "- 提炼核心价值、关键约束、跨章节的逻辑关联\n"
    "- 不得照搬任何一个章节的 summary 或 main_idea 原句\n"
    "- 每条 text 应能独立回答'这篇文档最重要的要点是什么'\n"
    "section_id 仍须对应输入中的 id。请重新生成。"
)

_RETRY_DENSITY_INSUFFICIENT = (
    "\n\n【必须满足】doc_outline_insights 至少 {need} 条；"
    "不足则拆分要点、同一节允许多条（不同 text、同一 section_id），"
    "直至条数达标且每条仍有实质信息。"
)

_RETRY_DIVERSITY_LACKING = (
    "\n\n【多样性不足】你的 doc_outline_insights 中大量条目使用了相同的句式模板（如'提供了新的…''为理解…提供了…'），"
    "导致多条 insight 表达的是同一个意思。请从以下**不同角度**各写 1-3 条，确保角度多样性：\n"
    "1. **核心价值**：本文档最重要的内容/结论/规定是什么，对读者有什么直接价值\n"
    "2. **关键约束或条件**：有哪些必须遵守的限制、前提条件、适用范围\n"
    "3. **跨章节关联**：不同章节之间有什么因果关系、依赖关系、互补关系\n"
    "4. **潜在风险或注意事项**：忽略文档中的哪些要点可能导致问题\n"
    "禁止使用'提供了新的视角''为理解…提供了重要依据'等空洞套话。"
    "section_id 仍须对应输入中的 id。请重新生成。"
)


def _collect_section_keywords(items: list[dict[str, Any]]) -> list[str]:
    """从 section 增强结果中收集所有 keywords（去重保序），用于 doc 级 prompt 的排除提示。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        kws = item.get("keywords", [])
        if not isinstance(kws, list):
            continue
        for kw in kws:
            if isinstance(kw, str) and kw and kw not in seen:
                seen.add(kw)
                result.append(kw)
    return result


def _fallback_doc_payload() -> dict[str, Any]:
    return {
        "global_summary": "未生成",
        "global_keywords": [],
        "doc_outline_insights": [],
        "answerable_questions": [],
    }


def _doc_schema_dict() -> dict[str, Any]:
    return {
        "global_summary": "string",
        "global_keywords": ["string"],
        "doc_outline_insights": [{"text": "string", "section_id": "string"}],
        "answerable_questions": ["string"],
    }


def _target_insight_count(l1_n: int) -> int:
    """目标条数：小文档按节翻倍，大文档对齐旧版密度（≥12 且每节可多条），上限 22。"""
    if l1_n <= 0:
        return 0
    if l1_n <= 4:
        return max(l1_n * 2, 4)
    return min(max(l1_n, 12), 22)


def _is_section_summaries_for_doc(items: list[Any]) -> bool:
    """Doc 阶段输入是否为 L1 章节摘要列表（而非 map-reduce 中的局部 overview 列表）。"""
    if not items:
        return False
    x = items[0]
    return isinstance(x, dict) and "global_summary" not in x and x.get("id") is not None


def _count_doc_insights(payload: dict[str, Any]) -> int:
    ins = payload.get("doc_outline_insights")
    return len(ins) if isinstance(ins, list) else 0


def _peek_l1_len_from_json(section_summaries_json: str) -> int:
    try:
        arr = json.loads(section_summaries_json)
        if isinstance(arr, list):
            return len(arr)
    except Exception:
        pass
    return 1


def _normalize_doc_outline_insights(raw: Any) -> list[dict[str, str]]:
    """仅接受对象数组，元素须含 text 与 section_id（与 schema 一致）。"""
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        sid = str(item.get("section_id", "")).strip()
        if not text or not sid:
            continue
        out.append({"text": text, "section_id": sid})
    return out


def _finalize_doc_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """将模型输出收敛为 overview 四键，不读取任何别名或历史字段。"""
    if not isinstance(payload, dict):
        return _fallback_doc_payload()
    p = dict(payload)
    gs = str(p.get("global_summary", "")).strip()
    gk = p.get("global_keywords")
    if not isinstance(gk, list):
        gk = []
    gk = [str(x).strip() for x in gk if str(x).strip()]
    insights = _normalize_doc_outline_insights(p.get("doc_outline_insights"))
    aq = p.get("answerable_questions")
    if not isinstance(aq, list):
        aq = []
    aq = [str(x).strip() for x in aq if str(x).strip()]
    return {
        "global_summary": gs,
        "global_keywords": gk,
        "doc_outline_insights": insights,
        "answerable_questions": aq,
    }


def _normalize_outline_item(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"^\d+(?:\.\d+)*\s*\.?\s*", "", s)
    s = re.sub(r"^[ivxlcdm]+\.\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^[a-z]\.\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _outline_insight_texts(payload: dict[str, Any]) -> list[str]:
    insights = payload.get("doc_outline_insights")
    if not isinstance(insights, list):
        return []
    texts: list[str] = []
    for x in insights:
        if isinstance(x, dict):
            texts.append(str(x.get("text", "")))
    return texts


def _outline_looks_like_titles(payload: dict[str, Any], success_items: list[dict[str, Any]]) -> bool:
    outline = _outline_insight_texts(payload)
    if len(outline) < 2:
        return False

    outline_norm = [_normalize_outline_item(x) for x in outline if str(x).strip()]
    title_norm = {
        _normalize_outline_item(item.get("title", ""))
        for item in success_items
        if _normalize_outline_item(item.get("title", ""))
    }
    if len(outline_norm) < 2 or len(title_norm) < 2:
        return False

    overlap = sum(1 for item in outline_norm if item in title_norm)
    return overlap >= max(2, len(outline_norm) - 1)


def _substring_overlap_ratio(a: str, b: str) -> float:
    """计算 a 是否是 b 的子串/高度重叠。

    用较短文本的字符在较长文本中命中的比例来衡量。
    """
    if not a or not b:
        return 0.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if short in long:
        return 1.0
    # 用 n-gram jaccard 近似
    n = 4
    if len(short) < n:
        return 1.0 if short in long else 0.0
    short_grams = {short[i:i+n] for i in range(len(short) - n + 1)}
    long_grams = {long[i:i+n] for i in range(len(long) - n + 1)}
    if not short_grams:
        return 0.0
    return len(short_grams & long_grams) / len(short_grams)


def _outline_looks_like_summaries(payload: dict[str, Any], success_items: list[dict[str, Any]]) -> bool:
    """检测 doc_outline_insights 是否只是各节 summary/main_idea 的复述。

    如果 ≥60% 的 insight 条目与对应 section 的 summary 或 main_idea 高度重叠，
    则判定为 summary 退化。
    """
    outline = _outline_insight_texts(payload)
    if len(outline) < 3:
        return False

    insights = payload.get("doc_outline_insights", [])
    if not isinstance(insights, list):
        return False

    # 建立 section_id → {summary, main_idea} 查找表
    section_texts: dict[str, list[str]] = {}
    for item in success_items:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id", ""))
        if not sid:
            continue
        texts: list[str] = []
        s = str(item.get("summary", "")).strip()
        if s:
            texts.append(s)
        m = str(item.get("main_idea", "")).strip()
        if m:
            texts.append(m)
        if texts:
            section_texts[sid] = texts

    if not section_texts:
        return False

    overlap_count = 0
    for ins in insights:
        if not isinstance(ins, dict):
            continue
        ins_text = str(ins.get("text", "")).strip()
        ins_sid = str(ins.get("section_id", "")).strip()
        if not ins_text or not ins_sid:
            continue
        ref_texts = section_texts.get(ins_sid, [])
        for ref in ref_texts:
            if _substring_overlap_ratio(ins_text, ref) >= 0.65:
                overlap_count += 1
                break

    return overlap_count >= max(2, int(len(insights) * 0.6))


def _insights_lack_diversity(payload: dict[str, Any]) -> bool:
    """检测 doc_outline_insights 是否语义同质化——多条 insight 表达同一含义。

    判定标准：如果 ≥60% 的 insight 包含"提供了""为…提供"等相同句式模板，
    则视为同质化。这是通用规则，不绑定特定文档主题。
    """
    insights = payload.get("doc_outline_insights")
    if not isinstance(insights, list) or len(insights) < 4:
        return False

    # 检测高频句式模板
    homogeneous_patterns = [
        "提供了", "为理解", "为评估", "为研究", "为未来",
        "新的视角", "新的方法", "新的工具", "新的见解",
        "重要依据", "重要基础", "重要支持",
    ]
    pattern_hit_count = 0
    for item in insights:
        text = str(item.get("text", ""))
        if any(pat in text for pat in homogeneous_patterns):
            pattern_hit_count += 1

    # 超过 60% 命中同质化模板
    return pattern_hit_count >= max(3, int(len(insights) * 0.6))


def _doc_output_max_tokens(insight_count: int) -> int:
    """动态估算 doc 级 LLM 输出所需 max_tokens。
    global_summary ~400 + global_keywords ~120 + answerable_questions ~300
    + doc_outline_insights(每条 ~80 tokens) + JSON 结构开销 ~100
    """
    return max(2000, 400 + 120 + insight_count * 80 + 300 + 100)


def _generate_doc_payload(
    *,
    provider: Any,
    prompt: str,
    schema_hint: str,
    success_items: list[dict[str, Any]],
    id_context: list[dict[str, Any]] | None = None,
    max_tokens: int = 2500,
) -> dict[str, Any]:
    """id_context: 用于 section_id 归一化与标题检测的 L1 章节列表；map-reduce 全局合并时传原始 L1，success_items 可为局部 overview。"""
    ic = id_context if id_context is not None else success_items
    payload = provider.generate_json(
        prompt=prompt,
        system_prompt=_DOC_SYSTEM_PROMPT,
        schema_hint=schema_hint,
        max_tokens=max_tokens,
    )
    payload = _finalize_doc_payload(payload)
    if _outline_looks_like_titles(payload, ic):
        logger.warning("[DocEnhanceStage] doc_outline_insights degenerated to titles, retrying")
        payload = provider.generate_json(
            prompt=prompt + _RETRY_TITLES_DEGENERATED,
            system_prompt=_DOC_SYSTEM_PROMPT,
            schema_hint=schema_hint,
            max_tokens=max_tokens,
        )
        payload = _finalize_doc_payload(payload)
    if _outline_looks_like_summaries(payload, ic):
        logger.warning("[DocEnhanceStage] doc_outline_insights degenerated to section summaries, retrying")
        payload = provider.generate_json(
            prompt=prompt + _RETRY_SUMMARIES_DEGENERATED,
            system_prompt=_DOC_SYSTEM_PROMPT,
            schema_hint=schema_hint,
            max_tokens=max_tokens,
        )
        payload = _finalize_doc_payload(payload)
    need: int | None = None
    if ic and _is_section_summaries_for_doc(ic):
        need = _target_insight_count(len(ic))
    if need is not None and need > 0 and _count_doc_insights(payload) < need:
        logger.warning(
            f"[DocEnhanceStage] doc_outline_insights sparse "
            f"({_count_doc_insights(payload)} < {need}), retry once for density"
        )
        payload = provider.generate_json(
            prompt=prompt + _RETRY_DENSITY_INSUFFICIENT.format(need=need),
            system_prompt=_DOC_SYSTEM_PROMPT,
            schema_hint=schema_hint,
            max_tokens=max_tokens,
        )
        payload = _finalize_doc_payload(payload)
    # 多样性检测：insights 是否语义同质化
    if _insights_lack_diversity(payload):
        logger.warning("[DocEnhanceStage] doc_outline_insights lack diversity, retrying with angle hints")
        payload = provider.generate_json(
            prompt=prompt + _RETRY_DIVERSITY_LACKING,
            system_prompt=_DOC_SYSTEM_PROMPT,
            schema_hint=schema_hint,
            max_tokens=max_tokens,
        )
        payload = _finalize_doc_payload(payload)
    return payload


def _estimate_doc_total_tokens(
    *,
    token_counter: TokenCounter,
    language: str,
    section_summaries_json: str,
) -> int:
    l1n = _peek_l1_len_from_json(section_summaries_json)
    prompt = render_prompt(
        "doc_summary.j2",
        {
            "language": language,
            "style": "retrieval_dense",
            "l1_section_count": l1n,
            "min_insights": _target_insight_count(l1n),
            "section_summaries": section_summaries_json,
            "section_keywords_to_avoid": "",
        },
    )
    system_prompt = _DOC_SYSTEM_PROMPT
    schema_hint = json.dumps(_doc_schema_dict(), ensure_ascii=False)
    output_reserve = 900
    return token_counter.count(system_prompt) + token_counter.count(prompt) + token_counter.count(schema_hint) + output_reserve


def _collect_success_descendants(item: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if item.get("status") == "success":
        item_copy = dict(item)
        item_copy.pop("subsections", None)
        out.append(item_copy)
    subs = item.get("subsections")
    if isinstance(subs, list):
        for sub in subs:
            if isinstance(sub, dict):
                out.extend(_collect_success_descendants(sub))
    return out


def run_doc_enhance_stage(
    *,
    provider: Any,
    section_results: list[dict[str, Any]],
    language: str,
    model_reference: str,
    available_input_tokens: int,
    token_counter: TokenCounter | None = None,
) -> dict[str, Any]:
    """
    DocEnhanceStage：只使用 L1 节点的增强结果（来自 section_results 顶层列表），
    并去掉 subsections，减少 prompt 噪音与 token 消耗。
    """
    success_items: list[dict[str, Any]] = []
    for item in section_results:
        if item.get("status") == "success":
            item_copy = dict(item)
            item_copy.pop("subsections", None)
            success_items.append(item_copy)
            continue

        # 父节点失败但子节点成功：退化为使用成功子树
        descendant_success = _collect_success_descendants(item)
        if descendant_success:
            merged_summary = "\n".join(
                [str(x.get("summary", "")).strip() for x in descendant_success if str(x.get("summary", "")).strip()]
            ).strip()
            merged_keywords: list[str] = []
            for x in descendant_success:
                kws = x.get("keywords", [])
                if isinstance(kws, list):
                    for kw in kws:
                        if isinstance(kw, str) and kw and kw not in merged_keywords:
                            merged_keywords.append(kw)
            success_items.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "page_range": item.get("page_range"),
                    "refs": item.get("refs", []),
                    "status": "success",
                    "summary": merged_summary or "子章节已成功生成摘要",
                    "keywords": merged_keywords[:10] or ["fallback"],
                    "main_idea": merged_summary or "子章节汇总",
                }
            )

    if not success_items:
        logger.info("[DocEnhanceStage] no success sections, skipping")
        return _fallback_doc_payload()

    l1_total = len(success_items)
    logger.info(f"[DocEnhanceStage] start: success_sections={l1_total}")

    # 收集所有 section keywords，作为 global_keywords 的排除提示
    _all_sec_kws = _collect_section_keywords(success_items)
    _sec_kws_str = ", ".join(_all_sec_kws) if _all_sec_kws else ""

    # 复用传入的 TokenCounter，或新建
    if token_counter is None:
        token_counter = TokenCounter.from_model(model_reference)

    # Mode A / B：token overflow 时做分组 map-reduce
    doc_input = json.dumps(success_items, ensure_ascii=False)
    total_tokens = _estimate_doc_total_tokens(
        token_counter=token_counter,
        language=language,
        section_summaries_json=doc_input,
    )
    single_limit = int(available_input_tokens * 0.9)
    logger.debug(
        f"[DocEnhanceStage] mode={'A' if total_tokens <= single_limit else 'B'} "
        f"estimated_tokens={total_tokens} limit={single_limit}"
    )

    if total_tokens <= single_limit:
        doc_prompt = render_prompt(
            "doc_summary.j2",
            {
                "language": language,
                "style": "retrieval_dense",
                "l1_section_count": l1_total,
                "min_insights": _target_insight_count(l1_total),
                "section_summaries": doc_input,
                "section_keywords_to_avoid": _sec_kws_str,
            },
        )
        doc_schema_hint = json.dumps(_doc_schema_dict(), ensure_ascii=False)
        try:
            return _generate_doc_payload(
                provider=provider,
                prompt=doc_prompt,
                schema_hint=doc_schema_hint,
                success_items=success_items,
                id_context=success_items,
                max_tokens=_doc_output_max_tokens(_target_insight_count(l1_total)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[DocEnhanceStage] doc enhance failed: {exc}")
            return _fallback_doc_payload()

    # Mode B：将 section_summaries 分组后做局部汇总再全局归并
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for item in success_items:
        projected = json.dumps(current + [item], ensure_ascii=False)
        projected_tokens = _estimate_doc_total_tokens(
            token_counter=token_counter,
            language=language,
            section_summaries_json=projected,
        )
        if current and projected_tokens > single_limit:
            groups.append(current)
            current = [item]
        else:
            current.append(item)
    if current:
        groups.append(current)

    local_payloads: list[dict[str, Any]] = []
    for g in groups:
        try:
            lg = len(g)
            local_prompt = render_prompt(
                "doc_summary.j2",
                {
                    "language": language,
                    "style": "retrieval_dense",
                    "l1_section_count": lg,
                    "min_insights": _target_insight_count(lg),
                    "section_summaries": json.dumps(g, ensure_ascii=False),
                    "section_keywords_to_avoid": _sec_kws_str,
                },
            )
            doc_schema_hint = json.dumps(_doc_schema_dict(), ensure_ascii=False)
            local_payload = _generate_doc_payload(
                provider=provider,
                prompt=local_prompt,
                schema_hint=doc_schema_hint,
                success_items=g,
                id_context=g,
                max_tokens=_doc_output_max_tokens(_target_insight_count(lg)),
            )
            local_payloads.append(local_payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[DocEnhanceStage] local group enhance failed: {exc}")

    # 全局归并
    global_prompt = render_prompt(
        "doc_summary.j2",
        {
            "language": language,
            "style": "retrieval_dense",
            "l1_section_count": l1_total,
            "min_insights": _target_insight_count(l1_total),
            "section_summaries": json.dumps(local_payloads, ensure_ascii=False),
            "section_keywords_to_avoid": _sec_kws_str,
        },
    )
    doc_schema_hint = json.dumps(_doc_schema_dict(), ensure_ascii=False)
    try:
        return _generate_doc_payload(
            provider=provider,
            prompt=global_prompt,
            schema_hint=doc_schema_hint,
            success_items=local_payloads,
            id_context=success_items,
            max_tokens=_doc_output_max_tokens(_target_insight_count(l1_total)),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[DocEnhanceStage] global doc enhance failed: {exc}")
        return _fallback_doc_payload()
