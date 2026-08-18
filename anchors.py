from __future__ import annotations

import os
import subprocess
from collections import defaultdict
from copy import deepcopy
from typing import Any

from cube_faces import image_index, image_meta
from geometry import rotation_for_bbox_center, rotation_for_face_center
from matterport import capture_anchors


DEFAULT_CONFIG: dict[str, Any] = {
    "workers": 1,
    "chunkSize": 1,
    "settleMs": 120,
    "persistentSession": True,
    "sessionIdleSeconds": 300,
    "headed": True,
}


def resolve_credentials(config: dict[str, Any], cube_faces: dict[str, Any]) -> tuple[str, str]:
    sdk_key = str(config.get("sdkKey") or os.getenv("MATTERPORT_SDK_KEY") or "")
    if not sdk_key:
        raise ValueError("MATTERPORT_SDK_KEY is required for 3D anchor generation.")
    model_id = str(config.get("modelId") or cube_faces.get("modelId") or cube_faces.get("model_id") or "")
    if not model_id:
        raise ValueError("cube_faces.json is missing modelId/model_id.")
    return model_id, sdk_key


def check_runtime() -> None:
    try:
        subprocess.run(["node", "-e", "import('playwright')"], check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise ValueError("Node.js is required for 3D anchor generation.") from error
    except subprocess.CalledProcessError as error:
        raise ValueError("Node package 'playwright' is required. Run `npm install` in this folder.") from error


def _task_instances(result: dict[str, Any]):
    for task in (result.get("prediction") or {}).get("tasks") or []:
        task_result = task.get("task_result")
        if isinstance(task_result, list):
            yield from (item for item in task_result if isinstance(item, dict))


def _base_3d_box(image: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": 0,
        "panoId": image["panoId"],
        "rotation": rotation_for_face_center(image),
        "anchorPosition": None,
        "stemVector": None,
    }


def _prediction_for_patch(result: dict[str, Any]) -> dict[str, Any]:
    prediction = result.get("prediction")
    if not isinstance(prediction, dict):
        prediction = {}
        result["prediction"] = prediction
    return prediction


def _prepare_backfill(
    raw_results: dict[str, dict[str, Any]],
    cube_faces: dict[str, Any],
    raw_image_names: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    index = image_index(cube_faces)
    base_boxes: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    box_count = 0
    for image_id, result in sorted(raw_results.items()):
        boxes_2d = (result.get("prediction") or {}).get("2d_bounding_boxes")
        boxes_2d = boxes_2d if isinstance(boxes_2d, list) else []
        box_count += len(boxes_2d)
        try:
            raw_image_name = raw_image_names.get(image_id)
            if not raw_image_name:
                raise ValueError("Missing rawImageName from preprocessed.jpg metadata.")
            image = image_meta(cube_faces, index, raw_image_name)
            base_box = _base_3d_box(image)
            base_boxes[image_id] = base_box
            for box in boxes_2d:
                rotation, center = rotation_for_bbox_center(box["coordinates"], image)
                bbox_id = int(box["id"])
                items.append({
                    "workItemId": f"{image_id}:{bbox_id}",
                    "s3Id": image_id,
                    "bbox2dId": bbox_id,
                    "bboxCenter": center,
                    "panoId": image["panoId"],
                    "rotation": rotation,
                    "storedRotation": base_box["rotation"],
                })
        except Exception as error:
            errors.append({"imageId": image_id, "error": str(error)})
    return base_boxes, items, errors, box_count


def patch_results(
    raw_results: dict[str, dict[str, Any]],
    captured_items: list[dict[str, Any]],
    base_boxes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in captured_items:
        grouped[str(item["s3Id"])].append(item)
    output: dict[str, dict[str, Any]] = {}
    for image_id, result in sorted(raw_results.items()):
        patched = deepcopy(result)
        items = grouped.get(image_id)
        if not items:
            if base_boxes and image_id in base_boxes:
                _prediction_for_patch(patched)["3d_bounding_boxes"] = [deepcopy(base_boxes[image_id])]
            output[image_id] = patched
            continue
        _prediction_for_patch(patched)["3d_bounding_boxes"] = [
            {
                "id": int(item["bbox2dId"]),
                "panoId": item["panoId"],
                "rotation": item["storedRotation"],
                "anchorPosition": item.get("anchorPosition"),
                "stemVector": item.get("stemVector"),
            }
            for item in sorted(items, key=lambda value: int(value["bbox2dId"]))
        ]
        for instance in _task_instances(patched):
            if instance.get("2d_bbox_id") is not None:
                instance["3d_bbox_id"] = int(instance["2d_bbox_id"])
        output[image_id] = patched
    return output


def build_anchor_metadata(
    captured_items: list[dict[str, Any]],
    raw_image_names: dict[str, str],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for item in sorted(captured_items, key=lambda value: (str(value["s3Id"]), int(value["bbox2dId"]))):
        surface_normal = item.get("surfaceNormal")
        if not isinstance(surface_normal, dict):
            continue
        image_id = str(item["s3Id"])
        bbox_id = int(item["bbox2dId"])
        payload = metadata.setdefault(image_id, {
            "schemaVersion": 1,
            "imageId": image_id,
            "rawImageName": raw_image_names.get(image_id),
            "boxes": {},
        })
        payload["boxes"][str(bbox_id)] = {
            "bbox2dId": bbox_id,
            "bbox3dId": bbox_id,
            "surfaceNormal": {axis: float(surface_normal[axis]) for axis in ("x", "y", "z")},
        }
    return metadata


def backfill_with_metadata(
    raw_results: dict[str, dict[str, Any]],
    cube_faces: dict[str, Any],
    raw_image_names: dict[str, str],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    base_boxes, work_items, errors, box_count = _prepare_backfill(raw_results, cube_faces, raw_image_names)
    if errors:
        raise ValueError(f"Could not prepare 3D metadata: {errors[:5]}")
    if not work_items:
        if box_count:
            raise ValueError(f"No 3D anchor work items were prepared: {errors[:5]}")
        return patch_results(raw_results, [], base_boxes), {}
    model_id, sdk_key = resolve_credentials(config, cube_faces)
    check_runtime()
    captured_items = capture_anchors(work_items, model_id, sdk_key, {**DEFAULT_CONFIG, **config})
    successful = [item for item in captured_items if not item.get("error") and not item.get("skipped")]
    skipped = [item for item in captured_items if item.get("skipped")]
    if skipped:
        import logging

        logging.warning(
            "Skipped %s Matterport anchor(s): %s",
            len(skipped),
            [f"{item.get('workItemId')}: {item.get('skipReason', 'invalid hit')}" for item in skipped[:5]],
        )
    return patch_results(raw_results, successful, base_boxes), build_anchor_metadata(successful, raw_image_names)
