from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from pydantic import BaseModel, Field, ValidationError
except Exception:  # pragma: no cover - graceful fallback when pydantic unavailable
    BaseModel = object  # type: ignore
    Field = None  # type: ignore
    ValidationError = Exception


# ---------------------------------------------------------------------------
# LLM 输出 Schema（Pydantic）
# ---------------------------------------------------------------------------

class SectionEnhanceModel(BaseModel):  # type: ignore[misc]
    summary: str
    keywords: list[str]
    main_idea: str
    questions: list[str] = []


class DocOutlineInsightItem(BaseModel):  # type: ignore[misc]
    """全文提纲洞察项：每条必须绑定章节 id，便于多路召回与引用校验。"""

    text: str
    section_id: str


class DocEnhanceModel(BaseModel):  # type: ignore[misc]
    global_summary: str
    global_keywords: list[str]
    doc_outline_insights: list[DocOutlineInsightItem] = []
    answerable_questions: list[str] = []


# ---------------------------------------------------------------------------
# 流水线中间数据结构
# ---------------------------------------------------------------------------

@dataclass
class EnhanceUnit:
    """Block Indexer + LLM Section Planner 输出的增强单元（flat，无树结构）。

    section_id 在 LLM 规划通过三层校验后才分配（s001, s002, …）。
    block_refs 来自对应 seq_ids 范围内所有 TitleItem/BodyItem 的有序 union。
    """
    section_id: str                          # "s001", "s002", …
    title: str                               # LLM 给出或继承的单元标题
    seq_ids: list[int]                       # 覆盖的 TitleItem 枚举位置列表（连续）
    body_text: str                           # 拼接的正文文本（供增强 LLM 使用）
    block_refs: list[list]                   # 三元列表有序 union：[page_no_1based, block_no_1based, type_str]
    page_range: list[int]                    # [min_page_no, max_page_no]（1-based）

    # 插入锚点：覆盖范围内第一个原始 TitleItem.title_text，供 ComposeStage 定位 markdown 标题行
    # 默认空串：compose 回退到 title；fallback/no-title 场景允许为空
    anchor_title: str = ""

    # TokenEstimationStage 填写
    estimated_tokens: int = 0


@dataclass
class SectionNode:
    """
    [DEPRECATED] SectionBuildStage/TokenEstimationStage/SectionPlanStage 用的旧数据结构（层级化 section tree）。

    新流程使用 EnhanceUnit（flat list）替代。
    保留此类以兼容尚未迁移的调用方（如 stages_validate.py、stages_compose.py 等）。

    约束：
    - level=0 的 root 节点的 own_text 必须为空
    - own_text 只包含"当前节点直属文本"（不含子节点文本）
    - children 顺序与原文一致
    """

    section_id: str
    title: str
    level: int
    own_text: str
    page_range: list[int]
    block_refs: list[dict[str, Any]] = field(default_factory=list)
    children: list["SectionNode"] = field(default_factory=list)

    # TokenEstimationStage 增加字段
    estimated_tokens: int = 0  # only own_text
    subtree_tokens: int = 0  # own_text + all descendants

    # SectionPlanStage 增加字段（用于 EnhanceUnit 标识）
    source_section_ids: list[str] = field(default_factory=list)  # 被合并的原始 id 列表
    merge_type: str = "standalone"  # "standalone" | "merged"
    unit_title: str = ""  # LLM 给出的合并单元标题（merged 时使用）

    def subtree_chars(self) -> int:
        """计算子树总字符数（含自身和所有后代）。"""
        total = len(self.own_text)
        for c in self.children:
            total += c.subtree_chars()
        return total

    def count_nodes(self) -> int:
        return 1 + sum(c.count_nodes() for c in self.children)

    def leaf_sections(self) -> list["SectionNode"]:
        if not self.children:
            return [self]
        out: list["SectionNode"] = []
        for c in self.children:
            out.extend(c.leaf_sections())
        return out

    def max_depth(self) -> int:
        if not self.children:
            return self.level
        return max(c.max_depth() for c in self.children)

    def to_dict_minimal(self) -> dict[str, Any]:
        """调试用：避免递归太深时直接 json 序列化 SectionNode。"""
        return {
            "section_id": self.section_id,
            "title": self.title,
            "level": self.level,
            "own_text_len": len(self.own_text),
            "page_range": self.page_range,
            "estimated_tokens": self.estimated_tokens,
            "subtree_tokens": self.subtree_tokens,
        }


# ---------------------------------------------------------------------------
# 校验函数
# ---------------------------------------------------------------------------

def validate_section_payload(payload: dict[str, Any]) -> tuple[bool, str]:
    try:
        SectionEnhanceModel(**payload)  # type: ignore[call-arg]
    except Exception as exc:
        return False, str(exc)
    if not payload.get("summary", "").strip():
        return False, "missing_summary"
    if not payload.get("keywords"):
        return False, "missing_keywords"
    return True, ""


def validate_doc_payload(payload: dict[str, Any]) -> tuple[bool, str]:
    try:
        DocEnhanceModel(**payload)  # type: ignore[call-arg]
    except Exception as exc:
        return False, str(exc)
    if not str(payload.get("global_summary", "")).strip():
        return False, "missing_global_summary"
    kws = payload.get("global_keywords")
    if not isinstance(kws, list) or len(kws) < 3:
        return False, "global_keywords_too_few"
    return True, ""
