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


def render_prompt(template_name: str, context: dict[str, Any]) -> str:
    template_path = PROMPT_DIR / template_name
    if Environment is None or FileSystemLoader is None:
        # Minimal fallback renderer for MVP robustness.
        raw = template_path.read_text(encoding="utf-8")
        text = raw
        for key, value in context.items():
            text = text.replace("{{ " + key + " }}", str(value))
            text = text.replace("{{" + key + "}}", str(value))
        return text

    env = Environment(
        loader=FileSystemLoader(str(PROMPT_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_name)
    return template.render(**context)

