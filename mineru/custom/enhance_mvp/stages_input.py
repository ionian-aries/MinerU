from typing import Any


def run_input_stage(
    *,
    middle_json: dict[str, Any],
    content_list_v2: Any,
    pdf_file_name: str,
    process_mode: str,
) -> dict[str, Any]:
    if not isinstance(middle_json, dict) or not middle_json.get("pdf_info"):
        raise ValueError("Enhance InputStage failed: middle_json.pdf_info is empty")
    if content_list_v2 is None:
        raise ValueError("Enhance InputStage failed: content_list_v2 is empty")
    return {
        "middle_json": middle_json,
        "content_list_v2": content_list_v2,
        "pdf_file_name": pdf_file_name,
        "process_mode": process_mode,
    }

