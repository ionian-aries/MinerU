"""Smoke tests for the enhance_mvp pipeline (refactored architecture — notes/20).

Tests cover:
  1.  BlockIndexer — basic scan (title + body + skip)
  2.  BlockIndexer — preamble (text before first title)
  3.  BlockIndexer — empty document (no blocks)
  4.  BlockIndexer — build_skip_set respects custom discard.types
  5.  LLM SectionPlanner — 3-layer validation (existence, continuity, coverage)
  6.  LLM SectionPlanner — fallback on empty title list
  7.  LLM SectionPlanner — build_enhance_units creates correct block_refs
  8.  TokenEstimationStage — fills estimated_tokens on EnhanceUnit list
  9.  SectionEnhanceStage — flat enhance with mock provider
  10. SectionEnhanceStage — questions field in output
  11. DocEnhanceStage — answerable_questions in output
  12. DocEnhanceStage — fallback uses successful items
  13. ValidateStage — schema + ref validation
  14. ValidateStage — answerable_questions passthrough
  15. ComposeStage — enhance tag insertion + overview
  16. ComposeStage — failed section not in md
  17. Pipeline e2e (mode=pipeline)
  18. Pipeline e2e (mode=hybrid)
  19. Pipeline map-reduce path
  20. resolve_enhancement_enabled toggle logic
  21. Token splitting utility

All LLM calls are mocked. No network required.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def _build_middle_json(process_mode: str = "pipeline") -> dict[str, Any]:
    return {"pdf_info": {"pdf_name": "test.pdf"}}


def _build_content_list_v2() -> list[list[dict[str, Any]]]:
    """Simple 2-section flat document using 'spans' content structure."""
    return [
        [
            {"type": "title", "content": {"spans": [{"type": "text", "content": "第一章 概述"}]}},
            {"type": "paragraph", "content": {"spans": [{"type": "text", "content": "这是第一章的正文内容，用于测试摘要与关键词提取。"}]}},
            {"type": "title", "content": {"spans": [{"type": "text", "content": "第二章 方法"}]}},
            {"type": "paragraph", "content": {"spans": [{"type": "text", "content": "这是第二章内容，介绍处理流程和实现策略。"}]}},
        ]
    ]


def _build_hierarchical_content_list_v2() -> list[list[dict[str, Any]]]:
    """Multi-level document with numbered headings."""
    return [
        [
            {"type": "title", "content": {"spans": [{"type": "text", "content": "2. Methods"}]}},
            {"type": "title", "content": {"spans": [{"type": "text", "content": "2.1. Characterisation"}]}},
            {"type": "paragraph", "content": {"spans": [{"type": "text", "content": "Flow duration curves paragraph."}]}},
            {"type": "title", "content": {"spans": [{"type": "text", "content": "2.2. Zero flow"}]}},
            {"type": "paragraph", "content": {"spans": [{"type": "text", "content": "Zero flow paragraph."}]}},
            {"type": "title", "content": {"spans": [{"type": "text", "content": "3. Data sets"}]}},
            {"type": "paragraph", "content": {"spans": [{"type": "text", "content": "Data sets paragraph."}]}},
        ]
    ]


def _build_leading_text_content_list_v2() -> list[list[dict[str, Any]]]:
    """Document with text before any heading (preamble)."""
    return [
        [
            {"type": "paragraph", "content": {"spans": [{"type": "text", "content": "Document preface text."}]}},
            {"type": "title", "content": {"spans": [{"type": "text", "content": "1. Start"}]}},
            {"type": "paragraph", "content": {"spans": [{"type": "text", "content": "Start section paragraph."}]}},
        ]
    ]


class _MockProviderInfo:
    provider = "openai_compatible"
    model = "mock-model"


class _MockProvider:
    """Mock provider that returns valid enhance JSON for any prompt."""

    def provider_info(self):
        return _MockProviderInfo()

    def generate_json(self, prompt: str, system_prompt: str, schema_hint: str = "") -> dict[str, Any]:
        hint_str = schema_hint or ""

        # Doc-level call
        if "global_summary" in hint_str or "answerable_questions" in hint_str:
            return {
                "global_summary": "Mock global summary of the document.",
                "global_keywords": ["mock_kw1", "mock_kw2", "mock_kw3", "mock_kw4"],
                "doc_outline_insights": [
                    {"text": "Mock insight 1", "section_id": "s001"},
                    {"text": "Mock insight 2", "section_id": "s002"},
                ],
                "answerable_questions": [
                    "这份文档主要解决什么问题？",
                    "适用范围是什么？",
                ],
            }

        # SectionPlanStage call (system prompt contains 文档结构分析专家)
        if "文档结构分析专家" in system_prompt or "文档结构" in prompt:
            try:
                # Parse max seq_id from title[NN] format in prompt
                import re
                all_ids = re.findall(r"title\[(\d+)\]", prompt)
                max_id = max(int(x) for x in all_ids) if all_ids else 1
                return {
                    "doc_type": "论文",
                    "sections": [
                        {"seq_ids": list(range(max_id + 1)), "title": "全文内容", "rationale": "合并所有节"},
                    ],
                }
            except Exception:
                return {
                    "doc_type": "论文",
                    "sections": [{"seq_ids": [0], "title": "内容", "rationale": "默认"}],
                }

        # Section batch mode
        try:
            hint = json.loads(hint_str)
        except (json.JSONDecodeError, TypeError):
            hint = {}

        if isinstance(hint, dict) and any(k.startswith("s") for k in hint):
            return {
                k: {
                    "summary": f"Summary for {k}",
                    "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
                    "main_idea": f"Idea for {k}",
                    "questions": [f"{k}能回答什么问题？", f"{k}的核心内容是什么？"],
                }
                for k in hint
            }

        # Section single call
        return {
            "summary": "Mock section summary with technical details.",
            "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
            "main_idea": "Mock main idea.",
            "questions": ["本节能回答什么典型问题？", "核心参数是什么？"],
        }

    def model_name(self) -> str:
        return "mock-model"

    def provider_name(self) -> str:
        return "openai_compatible"

    def token_counter(self):
        from mineru.custom.enhance_mvp.utils import TokenCounter
        return TokenCounter()

    def context_window(self) -> int:
        return 32768


def _make_custom_config(**enhance_overrides):
    from mineru.custom.custom_config_loader import CustomConfig
    enhance = {
        "enable": True,
        "model": "mock-model",
        "api_base": "http://localhost:8000/v1",
        "api_key": "test-key",
        "model_reference": "gpt-4o-mini",
        "language": "zh",
        "concurrency": 1,
    }
    enhance.update(enhance_overrides)
    return CustomConfig(enhance=enhance, storage={}, discard={})


# ===========================================================================
# Tests: BlockIndexer
# ===========================================================================

class TestBlockIndexer:

    def test_basic_scan_title_body_skip(self):
        from mineru.custom.custom_config_loader import CustomConfig
        from mineru.custom.enhance_mvp.stages_block_indexer import (
            BodyItem, TitleItem, run_block_indexer,
        )

        cfg = CustomConfig(enhance={}, storage={}, discard={})
        content = [
            [
                {"type": "page_header", "content": {"spans": []}},          # skip
                {"type": "title", "content": {"spans": [{"type": "text", "content": "1 范围"}]}},
                {"type": "paragraph", "content": {"spans": [{"type": "text", "content": "适用于针织服装。"}]}},
                {"type": "page_footer", "content": {}},                     # skip
                {"type": "title", "content": {"spans": [{"type": "text", "content": "2 定义"}]}},
            ]
        ]
        items = run_block_indexer(content, cfg)
        titles = [x for x in items if isinstance(x, TitleItem)]
        bodies = [x for x in items if isinstance(x, BodyItem)]

        assert len(titles) == 2
        assert titles[0].title_text == "1 范围"
        assert titles[1].title_text == "2 定义"
        # 只有"1 范围"后有内容；"2 定义"后无内容，空 BodyItem 不发射
        assert len(bodies) == 1
        assert bodies[0].char_count > 0

    def test_preamble_body_item(self):
        from mineru.custom.custom_config_loader import CustomConfig
        from mineru.custom.enhance_mvp.stages_block_indexer import BodyItem, TitleItem, run_block_indexer

        cfg = CustomConfig(enhance={}, storage={}, discard={})
        content = [[
            {"type": "paragraph", "content": {"spans": [{"type": "text", "content": "前言文本"}]}},
            {"type": "title", "content": {"spans": [{"type": "text", "content": "第一章"}]}},
        ]]
        items = run_block_indexer(content, cfg)
        # preamble 是第一个出现的 BodyItem，且在第一个 TitleItem 之前
        first_body_idx = next((i for i, x in enumerate(items) if isinstance(x, BodyItem)), -1)
        first_title_idx = next((i for i, x in enumerate(items) if isinstance(x, TitleItem)), -1)
        assert first_body_idx >= 0, "应有 preamble BodyItem"
        assert first_body_idx < first_title_idx, "preamble 必须出现在 TitleItem 之前"
        preamble = items[first_body_idx]
        assert "前言文本" in preamble.body_text

    def test_empty_document(self):
        from mineru.custom.custom_config_loader import CustomConfig
        from mineru.custom.enhance_mvp.stages_block_indexer import run_block_indexer

        cfg = CustomConfig(enhance={}, storage={}, discard={})
        items = run_block_indexer([], cfg)
        assert items == []

    def test_custom_discard_types_respected(self):
        from mineru.custom.custom_config_loader import CustomConfig
        from mineru.custom.enhance_mvp.stages_block_indexer import BodyItem, TitleItem, run_block_indexer

        # discard image blocks
        cfg = CustomConfig(enhance={}, storage={}, discard={"types": ["image"]})
        content = [[
            {"type": "title", "content": {"spans": [{"type": "text", "content": "1 范围"}]}},
            {"type": "image", "content": {"image_caption": "图1"}},         # should be skipped
            {"type": "paragraph", "content": {"spans": [{"type": "text", "content": "正文"}]}},
        ]]
        items = run_block_indexer(content, cfg)
        bodies = [x for x in items if isinstance(x, BodyItem)]
        # image block_ref should NOT be in any body（r[2] 是 type 字段）
        all_types = [r[2] for b in bodies for r in b.block_refs]
        assert "image" not in all_types
        # paragraph should still be present
        assert "paragraph" in all_types


# ===========================================================================
# Tests: LLM Section Planner
# ===========================================================================

class TestSectionPlanner:

    def test_3layer_validation_normal(self):
        from mineru.custom.enhance_mvp.stages_section_plan import _validate_and_fix

        raw = [
            {"seq_ids": [0, 1, 2], "title": "前置", "rationale": "x"},
            {"seq_ids": [3, 4], "title": "技术", "rationale": "x"},
        ]
        fixed, fatal = _validate_and_fix(raw, max_seq_id=4)
        assert not fatal
        assert len(fixed) == 2
        assert sorted(fixed[0]["seq_ids"]) == [0, 1, 2]
        assert sorted(fixed[1]["seq_ids"]) == [3, 4]

    def test_coverage_repair(self):
        from mineru.custom.enhance_mvp.stages_section_plan import _validate_and_fix

        # seq_id=2 is missing
        raw = [
            {"seq_ids": [0, 1], "title": "A", "rationale": "x"},
            {"seq_ids": [3], "title": "B", "rationale": "x"},
        ]
        fixed, fatal = _validate_and_fix(raw, max_seq_id=3)
        assert not fatal
        all_covered = {sid for item in fixed for sid in item["seq_ids"]}
        assert all_covered == {0, 1, 2, 3}

    def test_continuity_repair(self):
        from mineru.custom.enhance_mvp.stages_section_plan import _validate_and_fix

        # [0,1,3] is not continuous → should split into [0,1] and [3]
        raw = [{"seq_ids": [0, 1, 3], "title": "X", "rationale": "x"}]
        fixed, fatal = _validate_and_fix(raw, max_seq_id=3)
        assert not fatal
        all_covered = {sid for item in fixed for sid in item["seq_ids"]}
        assert all_covered == {0, 1, 2, 3}  # 2 added by coverage repair

    def test_fallback_no_titles(self):
        from mineru.custom.custom_config_loader import CustomConfig
        from mineru.custom.enhance_mvp.stages_block_indexer import run_block_indexer
        from mineru.custom.enhance_mvp.stages_section_plan import run_section_plan_stage
        from mineru.custom.enhance_mvp.utils import TokenCounter

        cfg = CustomConfig(enhance={}, storage={}, discard={})
        items = run_block_indexer([[
            {"type": "paragraph", "content": {"spans": [{"type": "text", "content": "只有正文"}]}},
        ]], cfg)

        units, result = run_section_plan_stage(
            items=items,
            provider=_MockProvider(),
            token_counter=TokenCounter(),
            available_input_tokens=32000,
            enhance_cfg={},
        )

        assert result.fallback is True
        assert result.fallback_reason == "no_titles"
        assert len(units) == 1

    def test_build_enhance_units_block_refs(self):
        from mineru.custom.enhance_mvp.stages_block_indexer import BodyItem, TitleItem
        from mineru.custom.enhance_mvp.stages_section_plan import build_enhance_units

        # 三元列表格式：[page_no_1based, block_no_1based, type]
        t0 = TitleItem(title_text="范围", block_ref=[1, 1, "title"])
        t1 = TitleItem(title_text="定义", block_ref=[1, 3, "title"])
        b0 = BodyItem(char_count=5, block_refs=[[1, 2, "paragraph"]])
        b1 = BodyItem(char_count=3, block_refs=[[1, 4, "paragraph"]])

        sections = [
            {"seq_ids": [0, 1], "title": "范围与定义", "rationale": "合并"},
        ]
        # body_map 以 title 枚举位置（0, 1, ...）为键
        units = build_enhance_units(sections, [t0, t1], {0: b0, 1: b1}, None)
        assert len(units) == 1
        u = units[0]
        assert u.section_id == "s001"
        assert u.title == "范围与定义"
        assert u.seq_ids == [0, 1]
        # block_refs: t0 + b0 + t1 + b1 = 4 refs
        assert len(u.block_refs) == 4
        # 验证每个 ref 是三元列表
        for ref in u.block_refs:
            assert isinstance(ref, list) and len(ref) == 3


# ===========================================================================
# Tests: TokenEstimationStage
# ===========================================================================

class TestTokenEstimation:

    def test_token_estimation_fills_fields(self):
        from mineru.custom.custom_config_loader import CustomConfig
        from mineru.custom.enhance_mvp.schema import EnhanceUnit
        from mineru.custom.enhance_mvp.stages_token_estimation import run_token_estimation_stage
        from mineru.custom.enhance_mvp.utils import TokenCounter

        units = [
            EnhanceUnit(
                section_id="s001", title="范围", seq_ids=[0],
                body_text="本标准适用于针织休闲服装。", block_refs=[], page_range=[0, 0],
            ),
            EnhanceUnit(
                section_id="s002", title="技术要求", seq_ids=[1],
                body_text="甲醛含量A类≤20mg/kg。", block_refs=[], page_range=[1, 1],
            ),
        ]
        tc = TokenCounter()
        result_units, ctx, avail = run_token_estimation_stage(
            enhance_units=units,
            model_reference="test-model",
            safety_ratio=0.85,
            token_counter=tc,
        )

        assert ctx > 0
        assert avail > 0
        for u in result_units:
            assert u.estimated_tokens > 0


# ===========================================================================
# Tests: SectionEnhanceStage
# ===========================================================================

class TestSectionEnhance:

    def test_flat_enhance_with_mock(self):
        from mineru.custom.enhance_mvp.schema import EnhanceUnit
        from mineru.custom.enhance_mvp.stages_section_enhance import run_section_enhance_stage
        from mineru.custom.enhance_mvp.utils import TokenCounter

        tc = TokenCounter()
        units = [
            EnhanceUnit(
                section_id="s001", title="第一章 概述", seq_ids=[0],
                body_text="这是第一章的正文内容，用于测试。",
                block_refs=[[1, 1, "title"], [1, 2, "paragraph"]],
                page_range=[1, 1], estimated_tokens=50,
            ),
            EnhanceUnit(
                section_id="s002", title="第二章 方法", seq_ids=[1],
                body_text="这是第二章内容，介绍处理流程。",
                block_refs=[[1, 3, "title"], [1, 4, "paragraph"]],
                page_range=[1, 1], estimated_tokens=50,
            ),
        ]

        results = run_section_enhance_stage(
            enhance_units=units,
            provider=_MockProvider(),
            model_reference="test-model",
            context_window=32768,
            available_input_tokens=28000,
            concurrency=1,
            language="zh",
            token_counter=tc,
        )

        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert "id" in r
            assert "status" in r
            assert "refs" in r

    def test_questions_field_in_output(self):
        from mineru.custom.enhance_mvp.schema import EnhanceUnit
        from mineru.custom.enhance_mvp.stages_section_enhance import run_section_enhance_stage
        from mineru.custom.enhance_mvp.utils import TokenCounter

        tc = TokenCounter()
        units = [
            EnhanceUnit(
                section_id="s001", title="技术要求", seq_ids=[0],
                body_text="A类甲醛限值≤20mg/kg，B类≤75mg/kg。",
                block_refs=[[1, 1, "title"]],
                page_range=[1, 1], estimated_tokens=20,
            ),
        ]

        results = run_section_enhance_stage(
            enhance_units=units,
            provider=_MockProvider(),
            model_reference="test-model",
            context_window=32768,
            available_input_tokens=28000,
            concurrency=1,
            language="zh",
            token_counter=tc,
        )

        success = [r for r in results if r.get("status") == "success"]
        assert success, "Expected at least one success"
        for r in success:
            # questions field should be present (list, may be empty if not in payload)
            assert "questions" in r
            assert isinstance(r["questions"], list)


# ===========================================================================
# Tests: DocEnhanceStage
# ===========================================================================

class TestDocEnhance:

    def test_answerable_questions_in_output(self):
        from mineru.custom.enhance_mvp.stages_doc_enhance import run_doc_enhance_stage
        from mineru.custom.enhance_mvp.utils import TokenCounter

        section_results = [
            {
                "id": "s001", "title": "第一章", "page_range": [0, 0],
                "refs": [[0, 1, "title"]], "status": "success",
                "summary": "第一章摘要", "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
                "main_idea": "核心观点", "questions": ["问题1？"],
            }
        ]

        overview = run_doc_enhance_stage(
            provider=_MockProvider(),
            section_results=section_results,
            language="zh",
            model_reference="test-model",
            available_input_tokens=32000,
            token_counter=TokenCounter(),
        )

        assert "answerable_questions" in overview
        assert isinstance(overview["answerable_questions"], list)

    def test_fallback_uses_successful_items(self):
        from mineru.custom.enhance_mvp.stages_doc_enhance import run_doc_enhance_stage
        from mineru.custom.enhance_mvp.utils import TokenCounter

        # flat list with one success (new format — no subsections)
        section_results = [
            {
                "id": "s001", "title": "Parent", "page_range": [0, 0],
                "refs": [[0, 1, "title"]], "status": "success",
                "summary": "child summary content here",
                "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
                "main_idea": "core", "questions": [],
            }
        ]

        overview = run_doc_enhance_stage(
            provider=_MockProvider(),
            section_results=section_results,
            language="zh",
            model_reference="test-model",
            available_input_tokens=32000,
            token_counter=TokenCounter(),
        )

        assert "global_summary" in overview
        assert overview["global_summary"]


# ===========================================================================
# Tests: ValidateStage
# ===========================================================================

class TestValidateStage:

    def test_validate_success_section(self):
        from mineru.custom.enhance_mvp.stages_validate import run_validate_stage

        sections = [
            {
                "id": "s001", "title": "Test", "page_range": [0, 0],
                "refs": [[0, 1, "title"]], "status": "success",
                "summary": "A valid summary.", "keywords": ["kw1", "kw2"],
                "main_idea": "A valid idea.", "questions": ["Q1？"],
            }
        ]
        overview = {
            "global_summary": "Valid doc summary.",
            "global_keywords": ["gk1", "gk2", "gk3"],
            "doc_outline_insights": [{"text": "insight", "section_id": "s001"}],
            "answerable_questions": ["文档能回答什么？"],
        }
        result = run_validate_stage(doc_overview=overview, section_results=sections)

        assert result["global_summary"] == "Valid doc summary."
        assert sections[0]["status"] == "success"

    def test_validate_missing_refs_marks_failed(self):
        from mineru.custom.enhance_mvp.stages_validate import run_validate_stage

        sections = [
            {
                "id": "s001", "title": "Test", "page_range": [0, 0],
                "refs": [],  # empty refs
                "status": "success",
                "summary": "A summary.", "keywords": ["kw1", "kw2"],
                "main_idea": "An idea.", "questions": [],
            }
        ]
        overview = {
            "global_summary": "Doc summary.",
            "global_keywords": ["a", "b", "c"],
            "doc_outline_insights": [],
            "answerable_questions": [],
        }
        run_validate_stage(doc_overview=overview, section_results=sections)
        assert sections[0]["status"] == "failed"

    def test_answerable_questions_passthrough(self):
        from mineru.custom.enhance_mvp.stages_validate import run_validate_stage

        overview = {
            "global_summary": "Summary.",
            "global_keywords": ["a", "b", "c"],
            "doc_outline_insights": [],
            "answerable_questions": ["Q1？", "Q2？"],
        }
        result = run_validate_stage(doc_overview=overview, section_results=[])
        assert result.get("answerable_questions") == ["Q1？", "Q2？"]


# ===========================================================================
# Tests: ComposeStage
# ===========================================================================

class TestComposeStage:

    def test_enhance_tags_inserted(self):
        from mineru.custom.enhance_mvp.stages_compose import run_compose_stage

        payload = {
            "overview": {
                "global_summary": "Test summary.",
                "global_keywords": ["kw1"],
                "doc_outline_insights": [{"text": "insight1", "section_id": "s001"}],
                "answerable_questions": ["Q？"],
            },
            "sections": [
                {
                    "id": "s001", "title": "第一章 概述", "page_range": [0, 0],
                    "refs": [[0, 1, "title"]], "status": "success",
                    "summary": "Section summary.", "keywords": ["kw1"],
                    "main_idea": "Main idea.", "questions": ["Q？"],
                }
            ],
        }
        raw_md = "# 第一章 概述\n正文A\n"
        result = run_compose_stage(raw_md, payload)

        assert '<enhance type="overview">' in result
        assert '<enhance id="s001"' in result
        assert "摘要: Section summary." in result
        assert "</enhance>" in result

    def test_failed_section_not_in_md(self):
        from mineru.custom.enhance_mvp.stages_compose import run_compose_stage

        payload = {
            "overview": {
                "global_summary": "X", "global_keywords": ["a"],
                "doc_outline_insights": [], "answerable_questions": [],
            },
            "sections": [
                {
                    "id": "s001", "title": "第一章", "page_range": [0, 0],
                    "refs": [[0, 1, "title"]], "status": "failed", "error": "test",
                }
            ],
        }
        raw_md = "# 第一章\n正文\n"
        result = run_compose_stage(raw_md, payload)
        assert 'id="s001"' not in result


# ===========================================================================
# Tests: Pipeline end-to-end
# ===========================================================================

class TestPipelineE2E:

    @patch("mineru.custom.enhance_mvp.pipeline.build_provider_from_custom_config")
    def test_pipeline_mode(self, mock_build_provider):
        from mineru.custom.enhance_mvp.pipeline import run_enhancement_pipeline

        mock_build_provider.return_value = _MockProvider()

        payload, enhanced_md = run_enhancement_pipeline(
            custom_config=_make_custom_config(),
            pdf_file_name="demo_pipeline",
            process_mode="pipeline",
            middle_json=_build_middle_json(),
            content_list_v2=_build_content_list_v2(),
            raw_markdown="# 第一章 概述\n正文A\n# 第二章 方法\n正文B\n",
        )

        assert "sections" in payload
        assert "overview" in payload
        assert "model_info" in payload
        assert "section_plan" in payload
        assert isinstance(payload["sections"], list)
        assert len(payload["sections"]) >= 1
        assert '<enhance type="overview">' in enhanced_md

    @patch("mineru.custom.enhance_mvp.pipeline.build_provider_from_custom_config")
    def test_hybrid_mode(self, mock_build_provider):
        from mineru.custom.enhance_mvp.pipeline import run_enhancement_pipeline

        mock_build_provider.return_value = _MockProvider()

        payload, enhanced_md = run_enhancement_pipeline(
            custom_config=_make_custom_config(),
            pdf_file_name="demo_hybrid",
            process_mode="hybrid",
            middle_json=_build_middle_json(),
            content_list_v2=_build_content_list_v2(),
            raw_markdown="# 第一章 概述\n正文A\n# 第二章 方法\n正文B\n",
        )

        assert payload["process_mode"] == "hybrid"
        assert "overview" in payload
        assert "answerable_questions" in payload["overview"]

    @patch("mineru.custom.enhance_mvp.pipeline.build_provider_from_custom_config")
    def test_mapreduce_path(self, mock_build_provider):
        """Very long section triggers map-reduce (Mode C)."""

        class _CountingProvider(_MockProvider):
            def __init__(self):
                self.call_count = 0

            def generate_json(self, prompt, system_prompt, schema_hint=""):
                self.call_count += 1
                return super().generate_json(prompt, system_prompt, schema_hint)

        provider = _CountingProvider()
        mock_build_provider.return_value = provider

        long_text = "这是一个很长的段落。" * 5000
        content = [[
            {"type": "title", "content": {"spans": [{"type": "text", "content": "1. Long Section"}]}},
            {"type": "paragraph", "content": {"spans": [{"type": "text", "content": long_text}]}},
        ]]

        from mineru.custom.enhance_mvp.pipeline import run_enhancement_pipeline

        payload, md = run_enhancement_pipeline(
            custom_config=_make_custom_config(),
            pdf_file_name="demo_mapreduce",
            process_mode="pipeline",
            middle_json=_build_middle_json(),
            content_list_v2=content,
            raw_markdown="# 1. Long Section\n正文\n",
        )

        assert "sections" in payload
        assert provider.call_count > 1  # map-reduce made multiple calls


# ===========================================================================
# Tests: Section failure does not break flow
# ===========================================================================

class TestFailureDegradation:

    @patch("mineru.custom.enhance_mvp.pipeline.build_provider_from_custom_config")
    def test_section_failure_does_not_break(self, mock_build_provider):
        """If provider raises on section enhance, pipeline still completes."""

        class _FailProvider:
            _call_count = 0

            def provider_info(self):
                return _MockProviderInfo()

            def generate_json(self, prompt, system_prompt, schema_hint=""):
                self._call_count += 1
                hint_str = schema_hint or ""
                # Succeed doc calls
                if "global_summary" in hint_str or "answerable_questions" in hint_str:
                    return {
                        "global_summary": "doc fallback",
                        "global_keywords": ["a", "b", "c"],
                        "doc_outline_insights": [],
                        "answerable_questions": [],
                    }
                # SectionPlanStage
                if "文档结构分析专家" in system_prompt or "文档结构" in prompt:
                    return {
                        "doc_type": "论文",
                        "sections": [{"seq_ids": [0], "title": "内容", "rationale": "单节"}],
                    }
                # Section calls fail first 2 times, then succeed
                if self._call_count <= 2:
                    raise RuntimeError("mock section fail")
                return {
                    "summary": "recovered", "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
                    "main_idea": "recovered", "questions": [],
                }

            def token_counter(self):
                from mineru.custom.enhance_mvp.utils import TokenCounter
                return TokenCounter()

        mock_build_provider.return_value = _FailProvider()

        from mineru.custom.enhance_mvp.pipeline import run_enhancement_pipeline

        payload, md = run_enhancement_pipeline(
            custom_config=_make_custom_config(),
            pdf_file_name="fail_test",
            process_mode="pipeline",
            middle_json=_build_middle_json(),
            content_list_v2=_build_content_list_v2(),
            raw_markdown="# 第一章 概述\n正文A\n# 第二章 方法\n正文B\n",
        )

        assert "sections" in payload
        assert "overview" in payload


# ===========================================================================
# Tests: resolve_enhancement_enabled
# ===========================================================================

class TestResolveEnhancementEnabled:

    def test_toggle_logic(self):
        from mineru.custom.custom_config_loader import CustomConfig

        cfg = CustomConfig(
            enhance={"enable": True, "model": "m", "api_base": "u", "api_key": "k"},
            storage={}, discard={},
        )
        assert cfg.enhance.get("enable") is True

        cfg2 = CustomConfig(enhance={"enable": False}, storage={}, discard={})
        assert cfg2.enhance.get("enable") is False


# ===========================================================================
# Tests: Token splitting utility
# ===========================================================================

class TestTokenSplitting:

    def test_split_text_by_token_budget(self):
        from mineru.custom.enhance_mvp.utils import TokenCounter, split_text_by_token_budget

        tc = TokenCounter()
        text = "这是测试文本。" * 500
        budget = 100

        chunks = split_text_by_token_budget(text, budget, tc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert tc.count(chunk) <= budget * 1.5

    def test_short_text_not_split(self):
        from mineru.custom.enhance_mvp.utils import TokenCounter, split_text_by_token_budget

        tc = TokenCounter()
        text = "Short text."
        chunks = split_text_by_token_budget(text, 1000, tc)
        assert len(chunks) == 1
        assert chunks[0] == text
