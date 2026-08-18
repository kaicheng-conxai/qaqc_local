from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NmsReasoningItem:
    issue_id: str
    reasoning: str


@dataclass(frozen=True)
class NmsReasoningOutput:
    items: tuple[NmsReasoningItem, ...]


def reasoning_parser():
    from langchain_core.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field

    class NmsReasoningItemSchema(BaseModel):
        issue_id: str = Field(..., description="The unchanged anonymous issue ID from the input.")
        reasoning: str = Field(..., description="The final reasoning for this issue.")

    class NmsReasoningSchema(BaseModel):
        items: list[NmsReasoningItemSchema] = Field(
            ...,
            description="Exactly one final reasoning result for every input issue.",
        )

    NmsReasoningSchema.model_rebuild(
        _types_namespace={"NmsReasoningItemSchema": NmsReasoningItemSchema}
    )
    return PydanticOutputParser(pydantic_object=NmsReasoningSchema)


def normalize_reasoning_response(raw: Any) -> NmsReasoningOutput:
    if isinstance(raw, NmsReasoningOutput):
        return raw
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        raise ValueError("NMS AI reasoning response must be a JSON object.")
    raw_items = raw.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("NMS AI reasoning response must include an items array.")

    items = []
    for raw_item in raw_items:
        if hasattr(raw_item, "model_dump"):
            raw_item = raw_item.model_dump()
        if not isinstance(raw_item, dict):
            raise ValueError("Every NMS AI reasoning item must be a JSON object.")
        issue_id = raw_item.get("issue_id")
        reasoning = raw_item.get("reasoning")
        if not isinstance(issue_id, str) or not issue_id.strip():
            raise ValueError("Every NMS AI reasoning item must include a non-empty issue_id string.")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("Every NMS AI reasoning item must include a non-empty reasoning string.")
        items.append(NmsReasoningItem(issue_id=issue_id.strip(), reasoning=reasoning.strip()))
    return NmsReasoningOutput(items=tuple(items))
