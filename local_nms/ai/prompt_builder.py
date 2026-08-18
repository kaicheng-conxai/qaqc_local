from __future__ import annotations

import json
from typing import Any


def _tag_payload(tag: dict[str, Any], index: int, keeper_tag_id: str) -> dict[str, Any]:
    return {
        "detection_index": index,
        "is_keeper": tag.get("tagKey") == keeper_tag_id,
        "task": tag.get("taskName"),
        "reasoning": tag.get("reasoning"),
    }


def build_user_prompt(
    items: list[dict[str, Any]],
    prompts: dict[str, str],
) -> str:
    return json.dumps(
        {
            "operation_instructions": {
                "cleanup": prompts["cleanup"],
                "merge": prompts["merge"],
            },
            "issues": [
                {
                    "issue_id": item["issue_id"],
                    "operation": item["operation"],
                    "detections": [
                        _tag_payload(tag, index, item["keeper_key"])
                        for index, tag in enumerate(item["members"], start=1)
                    ],
                }
                for item in items
            ],
        },
        ensure_ascii=False,
    )
