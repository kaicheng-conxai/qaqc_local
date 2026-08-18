from __future__ import annotations

from copy import deepcopy
from typing import Any


def build_nms_input(
    image_id: str,
    source: dict[str, Any],
    processed_result: dict[str, Any],
    anchor_metadata: dict[str, Any],
    run_id: str,
    completed_at: str,
) -> dict[str, Any]:
    result = deepcopy(processed_result)
    boxes = (result.get("prediction") or {}).get("3d_bounding_boxes") or []
    metadata_boxes = anchor_metadata.get("boxes") or {}
    for box in boxes:
        if not isinstance(box, dict):
            continue
        metadata = metadata_boxes.get(str(box.get("id"))) or metadata_boxes.get(box.get("id"))
        if isinstance(metadata, dict) and isinstance(metadata.get("surfaceNormal"), dict):
            box["surfaceNormal"] = deepcopy(metadata["surfaceNormal"])
    return {
        "schemaVersion": 1,
        "imageId": image_id,
        "source": deepcopy(source),
        "result": result,
        "processing": {"runId": run_id, "completedAt": completed_at},
    }
