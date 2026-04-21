from mineru.utils.enum_class import BlockType, ContentTypeV2

# BlockType（middle_json）→ ContentTypeV2（content_list_v2）噪音类型映射。
# 唯一来源：backend 用 BlockType 做 should_discard，enhance pipeline 用 ContentTypeV2 做 skip。
NOISE_BLOCK_TO_CV2: dict[str, str] = {
    BlockType.HEADER:        ContentTypeV2.PAGE_HEADER,
    BlockType.FOOTER:        ContentTypeV2.PAGE_FOOTER,
    BlockType.PAGE_NUMBER:   ContentTypeV2.PAGE_NUMBER,
    BlockType.ASIDE_TEXT:    ContentTypeV2.PAGE_ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE: ContentTypeV2.PAGE_FOOTNOTE,
}

DEFAULT_DISCARD_TYPES = frozenset(NOISE_BLOCK_TO_CV2)


class DiscardPolicy:
    def __init__(self, discard_types: frozenset[str] | None = None):
        self.discard_types = discard_types or DEFAULT_DISCARD_TYPES

    @classmethod
    def from_discard_types(cls, discard_types_input) -> "DiscardPolicy":
        discard_types = _normalize_types(discard_types_input)
        return cls(discard_types=discard_types)

    def should_discard(self, block: dict) -> bool:
        block_type = block.get("type")
        return isinstance(block_type, str) and block_type in self.discard_types


def _normalize_types(values) -> frozenset[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        normalized_values = [item.strip() for item in values.split(",") if item.strip()]
    elif isinstance(values, list):
        normalized_values = values
    else:
        raise ValueError("discard_types must be a comma-separated string or list of BlockType values.")

    if not normalized_values:
        return None

    valid_types = _get_valid_block_types()
    invalid_values = [
        item for item in normalized_values
        if not isinstance(item, str) or item not in valid_types
    ]
    if invalid_values:
        raise ValueError(
            "discard_types contains invalid BlockType values: "
            + ", ".join(map(str, invalid_values))
        )
    return frozenset(normalized_values)


def _get_valid_block_types() -> set[str]:
    return {
        value
        for key, value in vars(BlockType).items()
        if not key.startswith("_") and isinstance(value, str)
    }
