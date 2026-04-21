from __future__ import annotations

from pathlib import Path
from typing import Any

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except Exception:  # pragma: no cover - fallback if jinja2 unavailable
    Environment = None  # type: ignore
    FileSystemLoader = None  # type: ignore
    StrictUndefined = None  # type: ignore


def render_prompt(
    template_name: str,
    context: dict[str, Any],
    prompt_dir: Path | None = None,
) -> str:
    """渲染 Jinja2 prompt 模板。

    Args:
        template_name: 模板文件名（如 "section_summary.j2"）。
        context: 模板变量字典。
        prompt_dir: 模板目录路径。None 表示使用内置 PROMPT_DIR。
            可通过 manifest.yaml._prompt_dir 传入外部目录，实现热插拔自定义模板。
    """
    base_dir = prompt_dir or PROMPT_DIR
    template_path = base_dir / template_name

    if Environment is None or FileSystemLoader is None:
        # Minimal fallback renderer for MVP robustness.
        raw = template_path.read_text(encoding="utf-8")
        text = raw
        for key, value in context.items():
            text = text.replace("{{ " + key + " }}", str(value))
            text = text.replace("{{" + key + "}}", str(value))
        return text

    env = Environment(
        loader=FileSystemLoader(str(base_dir)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_name)
    return template.render(**context)

