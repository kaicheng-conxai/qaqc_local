"""Non-maximum suppression for duplicate QAQC issue tags."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from local_nms.cube_faces import parse_image_name
from local_nms.types import IssueInfo
from local_nms.ai import reasoning as nms_ai_reasoning
from local_nms.issue_ids import format_issue_id


SEVERITY_RANK = {"major": 0, "medium": 1, "minor": 2}
DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "distance": 0.5,
    "veryCloseDistance": 0.1,
    "clusterMode": "greedy_direct",
    "groupBy": "task_object_type",
    "keeperSelection": "cluster_middle",
    "taskDistances": {},
}


def _prediction(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("prediction")
    return value if isinstance(value, dict) else {}


def _three_d_boxes(result: dict[str, Any]) -> list[dict[str, Any]]:
    value = _prediction(result).get("3d_bounding_boxes")
    return value if isinstance(value, list) else []


def _task_instances(result: dict[str, Any]):
    for task in _prediction(result).get("tasks", []) or []:
        task_result = task.get("task_result")
        if not isinstance(task_result, list):
            continue
        for instance in task_result:
            if isinstance(instance, dict):
                yield task, instance


def _severity_key(value: object) -> str:
    return str(value or "").lower()


def _task_code(task_name: object) -> str:
    return str(task_name or "").split(":", 1)[0].strip() or "unknown"


def _object_type(reasoning: object) -> str | None:
    first_section = str(reasoning or "").split(";", 1)[0].strip()
    if "-" not in first_section:
        return None
    object_type, damage_type = first_section.split("-", 1)
    if not object_type.strip() or not damage_type.strip():
        return None
    return " ".join(object_type.split()).casefold()


def _point(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    coords = (value.get("x"), value.get("y"), value.get("z"))
    if not all(isinstance(coord, (int, float)) for coord in coords):
        return None
    return {"x": float(coords[0]), "y": float(coords[1]), "z": float(coords[2])}


def _vector2d(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    coords = (value.get("x"), value.get("y"))
    if not all(isinstance(coord, (int, float)) for coord in coords):
        return None
    return {"x": float(coords[0]), "y": float(coords[1])}


def _distance(left: dict[str, float], right: dict[str, float]) -> float:
    return math.dist((left["x"], left["y"], left["z"]), (right["x"], right["y"], right["z"]))


def _normalized_vector(value: Any) -> tuple[float, float, float] | None:
    point = _point(value)
    if not point:
        return None
    length = math.sqrt(point["x"] ** 2 + point["y"] ** 2 + point["z"] ** 2)
    if not length:
        return None
    return (point["x"] / length, point["y"] / length, point["z"] / length)


def _vectors_match(left: Any, right: Any, threshold: float | None, *, absolute: bool = False) -> bool:
    if threshold is None:
        return True
    left_vector = _normalized_vector(left)
    right_vector = _normalized_vector(right)
    if left_vector is None or right_vector is None:
        return False
    dot = sum(a * b for a, b in zip(left_vector, right_vector))
    return abs(dot) >= threshold if absolute else dot >= threshold


def _surface_plane_distance(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    left_normal = _normalized_vector(left.get("surfaceNormal"))
    right_normal = _normalized_vector(right.get("surfaceNormal"))
    if left_normal is None or right_normal is None:
        return None
    left_anchor = left["anchorPosition"]
    right_anchor = right["anchorPosition"]
    delta = (
        float(right_anchor["x"]) - float(left_anchor["x"]),
        float(right_anchor["y"]) - float(left_anchor["y"]),
        float(right_anchor["z"]) - float(left_anchor["z"]),
    )
    left_distance = abs(sum(delta[index] * left_normal[index] for index in range(3)))
    right_distance = abs(sum(delta[index] * right_normal[index] for index in range(3)))
    return max(left_distance, right_distance)


def _surfaces_match(
    left: dict[str, Any],
    right: dict[str, Any],
    normal_dot: float | None,
    plane_distance: float | None,
) -> bool:
    if not _vectors_match(left.get("surfaceNormal"), right.get("surfaceNormal"), normal_dot, absolute=True):
        return False
    if plane_distance is None:
        return True
    distance_value = _surface_plane_distance(left, right)
    return distance_value is not None and distance_value <= plane_distance


def _tag_key(tag: dict[str, Any]) -> str:
    return f"{tag.get('imageId')}:{tag.get('bbox2dId')}:{tag.get('taskId')}:{tag.get('instanceId')}"


def _tag_sort_key(tag: dict[str, Any]) -> tuple:
    return (
        tag.get("imageId") or "",
        tag.get("bbox2dId") if tag.get("bbox2dId") is not None else 999999,
        tag.get("taskId") if tag.get("taskId") is not None else 999999,
        tag.get("instanceId") if tag.get("instanceId") is not None else 999999,
    )


def _keeper_sort_key(tag: dict[str, Any]) -> tuple:
    return (
        SEVERITY_RANK.get(_severity_key(tag.get("class")), 9),
        -len(tag.get("reasoning") or ""),
        *_tag_sort_key(tag),
    )


def _cluster_centroid(tags: list[dict[str, Any]]) -> tuple[float, float, float]:
    return (
        sum(tag["anchorPosition"]["x"] for tag in tags) / len(tags),
        sum(tag["anchorPosition"]["y"] for tag in tags) / len(tags),
        sum(tag["anchorPosition"]["z"] for tag in tags) / len(tags),
    )


def _select_keeper(tags: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if mode == "severity":
        return sorted(tags, key=_keeper_sort_key)[0]
    if mode in {"cluster_middle", "middle", "centroid"}:
        centroid = _cluster_centroid(tags)
        return sorted(
            tags,
            key=lambda tag: (
                math.dist(
                    (tag["anchorPosition"]["x"], tag["anchorPosition"]["y"], tag["anchorPosition"]["z"]),
                    centroid,
                ),
                *_keeper_sort_key(tag),
            ),
        )[0]
    raise ValueError('nms keeperSelection must be "severity" or "cluster_middle".')


def _room_scope_key(tag: dict[str, Any], room_scope: str) -> str:
    if room_scope == "off":
        return "all_rooms"
    return f"{tag.get('floorId') or 'blank_floor'}:{tag.get('roomId') or 'blank_room'}"


def _nms_group_key(tag: dict[str, Any], group_by: str) -> str:
    if group_by == "all":
        return "all"
    if group_by == "trade":
        return tag.get("trade") or tag.get("taskName") or ""
    if group_by == "object_type":
        object_type = tag.get("objectType")
        return object_type or f"singleton:{_tag_key(tag)}"
    if group_by == "task_object_type":
        object_type = tag.get("objectType") or f"singleton:{_tag_key(tag)}"
        return f"task={tag.get('taskName') or ''}|objectType={object_type}"
    return tag.get("taskName") or ""


def _threshold_for_group(tags: list[dict[str, Any]], default_distance: float, task_distances: dict[str, Any]) -> float:
    if not tags:
        return default_distance
    task_name = tags[0].get("taskName")
    return float(task_distances.get(task_name, default_distance))


def _same_task(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_task_id = left.get("taskId")
    right_task_id = right.get("taskId")
    if left_task_id is not None and right_task_id is not None:
        return left_task_id == right_task_id

    left_code = _task_code(left.get("taskName"))
    right_code = _task_code(right.get("taskName"))
    return left_code != "unknown" and left_code == right_code


def _tags_match(left: dict[str, Any], right: dict[str, Any], threshold: float, settings: dict[str, Any]) -> bool:
    distance_value = _distance(left["anchorPosition"], right["anchorPosition"])
    very_close_distance = settings.get("veryCloseDistance")
    if very_close_distance is not None and _same_task(left, right) and distance_value <= float(very_close_distance):
        return True

    return (
        distance_value <= threshold
        and _vectors_match(left.get("stemVector"), right.get("stemVector"), settings.get("stemDot"))
        and _surfaces_match(
            left,
            right,
            settings.get("surfaceNormalDot"),
            settings.get("surfacePlaneDistance"),
        )
    )


def _union_clusters(tags: list[dict[str, Any]], threshold: float, settings: dict[str, Any]) -> list[list[int]]:
    parent = list(range(len(tags)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(tags)):
        for right in range(left + 1, len(tags)):
            if _tags_match(tags[left], tags[right], threshold, settings):
                union(left, right)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(tags)):
        clusters[find(index)].append(index)
    return [cluster for cluster in clusters.values() if len(cluster) > 1]


def _greedy_direct_clusters(tags: list[dict[str, Any]], threshold: float, settings: dict[str, Any]) -> list[list[int]]:
    remaining = sorted(range(len(tags)), key=lambda index: _keeper_sort_key(tags[index]))
    clusters = []
    while remaining:
        keeper_index = remaining.pop(0)
        matched = [index for index in remaining if _tags_match(tags[keeper_index], tags[index], threshold, settings)]
        if matched:
            clusters.append([keeper_index, *matched])
            matched_set = set(matched)
            remaining = [index for index in remaining if index not in matched_set]
    return clusters


def _cluster_group(tags: list[dict[str, Any]], threshold: float, settings: dict[str, Any]) -> list[list[int]]:
    mode = settings.get("clusterMode", "greedy_direct")
    if mode == "union":
        return _union_clusters(tags, threshold, settings)
    if mode == "greedy_direct":
        return _greedy_direct_clusters(tags, threshold, settings)
    raise ValueError('nms clusterMode must be "union" or "greedy_direct".')


def _task_matches(tag: dict[str, Any], tasks: list[str] | None) -> bool:
    return not tasks or _task_code(tag.get("taskName")) in tasks or tag.get("taskName") in tasks


def _config_passes(nms_config: dict[str, Any]) -> list[dict[str, Any]]:
    flat_defaults = {
        key: value
        for key, value in nms_config.items()
        if key not in {"enabled", "defaults", "passes", "faceFilter"}
    }
    defaults = {**flat_defaults, **(nms_config.get("defaults") or {})}
    configured_passes = nms_config.get("passes") or [{}]

    passes = []
    for index, nms_pass in enumerate(configured_passes, start=1):
        settings = {**defaults, **nms_pass}
        settings.setdefault("name", f"pass_{index}")
        settings.setdefault("tasks", None)
        settings.setdefault("distance", 0.5)
        settings.setdefault("veryCloseDistance", 0.1)
        settings.setdefault("taskDistances", {})
        settings.setdefault("groupBy", "task")
        settings.setdefault("roomScope", "auto")
        settings.setdefault("stemDot", None)
        settings.setdefault("clusterMode", "greedy_direct")
        settings.setdefault("keeperSelection", "cluster_middle")
        settings.setdefault("surfaceNormalDot", None)
        settings.setdefault("surfacePlaneDistance", None)
        passes.append(settings)
    return passes


def _run_pass(tags: list[dict[str, Any]], settings: dict[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
    duplicate_keys: set[str] = set()
    clusters = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for tag in tags:
        groups[(_room_scope_key(tag, settings["roomScope"]), _nms_group_key(tag, settings["groupBy"]))].append(tag)

    for (room_scope, group_name), group_tags in groups.items():
        threshold = _threshold_for_group(group_tags, float(settings["distance"]), settings.get("taskDistances") or {})
        for indexes in _cluster_group(group_tags, threshold, settings):
            cluster_tags = [group_tags[index] for index in indexes]
            keeper = _select_keeper(cluster_tags, settings.get("keeperSelection", "cluster_middle"))
            keeper_key = _tag_key(keeper)
            duplicate_keys.update(_tag_key(tag) for tag in cluster_tags if _tag_key(tag) != keeper_key)
            clusters.append(
                {
                    "passName": settings["name"],
                    "roomScope": room_scope,
                    "group": group_name,
                    "threshold": threshold,
                    "keeperSelection": settings.get("keeperSelection", "cluster_middle"),
                    "keeper": keeper,
                    "duplicates": [tag for tag in cluster_tags if _tag_key(tag) != keeper_key],
                    "tags": cluster_tags,
                }
            )
    return duplicate_keys, clusters


def _merge_pass_clusters(pass_clusters: list[dict[str, Any]]) -> tuple[set[str], list[dict[str, Any]]]:
    tags_by_key: dict[str, dict[str, Any]] = {}
    for cluster in pass_clusters:
        for tag in cluster.get("tags", []):
            tags_by_key.setdefault(_tag_key(tag), tag)

    parent = {key: key for key in tags_by_key}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for cluster in pass_clusters:
        keys = [_tag_key(tag) for tag in cluster.get("tags", []) if _tag_key(tag) in parent]
        if len(keys) < 2:
            continue
        for key in keys[1:]:
            union(keys[0], key)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, tag in tags_by_key.items():
        groups[find(key)].append(tag)

    source_clusters_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cluster in pass_clusters:
        keys = [_tag_key(tag) for tag in cluster.get("tags", []) if _tag_key(tag) in parent]
        if keys:
            source_clusters_by_root[find(keys[0])].append(cluster)

    duplicate_keys: set[str] = set()
    final_clusters = []
    for root, group_tags in groups.items():
        if len(group_tags) < 2:
            continue
        source_clusters = source_clusters_by_root[root]
        first_source = source_clusters[0]
        tags = sorted(group_tags, key=_tag_sort_key)
        keeper = _select_keeper(tags, first_source.get("keeperSelection", "cluster_middle"))
        keeper_key = _tag_key(keeper)
        duplicates = [tag for tag in tags if _tag_key(tag) != keeper_key]
        duplicate_keys.update(_tag_key(tag) for tag in duplicates)
        pass_names = []
        for source in source_clusters:
            if source.get("passName") not in pass_names:
                pass_names.append(source.get("passName"))
        final_clusters.append(
            {
                "passName": first_source.get("passName"),
                "passNames": pass_names,
                "roomScope": first_source.get("roomScope"),
                "group": first_source.get("group"),
                "threshold": first_source.get("threshold"),
                "keeper": keeper,
                "duplicates": duplicates,
                "tags": tags,
            }
        )

    final_clusters.sort(key=lambda cluster: min(_tag_sort_key(tag) for tag in cluster["tags"]))
    return duplicate_keys, final_clusters


def _run_all_passes(
    tags: list[dict[str, Any]], passes: list[dict[str, Any]]
) -> tuple[set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    all_pass_clusters = []
    summaries = []
    for settings in passes:
        eligible = [tag for tag in tags if _task_matches(tag, settings.get("tasks"))]
        pass_duplicate_keys, current_pass_clusters = _run_pass(eligible, settings)
        all_pass_clusters.extend(current_pass_clusters)
        summaries.append(
            {
                "name": settings["name"],
                "tasks": settings.get("tasks"),
                "distance": settings["distance"],
                "veryCloseDistance": settings.get("veryCloseDistance"),
                "taskDistances": settings.get("taskDistances") or {},
                "groupBy": settings["groupBy"],
                "roomScope": settings["roomScope"],
                "stemDot": settings.get("stemDot"),
                "clusterMode": settings.get("clusterMode"),
                "surfaceNormalDot": settings.get("surfaceNormalDot"),
                "surfacePlaneDistance": settings.get("surfacePlaneDistance"),
                "eligibleCount": len(eligible),
                "duplicateCount": len(pass_duplicate_keys),
                "clusterCount": len(current_pass_clusters),
            }
        )
    duplicate_keys, clusters = _merge_pass_clusters(all_pass_clusters)
    return duplicate_keys, clusters, summaries


def _summarize(tag: dict[str, Any], *, reasoning: str | None = None) -> dict[str, Any]:
    return {
        "tagKey": _tag_key(tag),
        "issueId": tag.get("issueId"),
        "imageId": tag.get("imageId"),
        "rawImageName": tag.get("rawImageName"),
        "bbox2dId": tag.get("bbox2dId"),
        "bbox3dId": tag.get("bbox3dId"),
        "taskId": tag.get("taskId"),
        "taskName": tag.get("taskName"),
        "objectType": tag.get("objectType"),
        "issueType": tag.get("issueType"),
        "class": tag.get("class"),
        "trade": tag.get("trade"),
        "roomId": tag.get("roomId"),
        "floorId": tag.get("floorId"),
        "faceDirection": tag.get("faceDirection"),
        "rotation": tag.get("rotation"),
        "anchorPosition": tag.get("anchorPosition"),
        "stemVector": tag.get("stemVector"),
        "reasoning": tag.get("reasoning") if reasoning is None else reasoning,
    }


def _issue_type(task: dict[str, Any], instance: dict[str, Any], issue_config: dict[str, Any]) -> str:
    source = issue_config.get("issueTypeSource", "task_code")
    if source == "class":
        return str(instance.get("instance_class") or "unknown")
    if source == "task_name":
        return str(task.get("task_name") or "unknown")
    return _task_code(task.get("task_name"))


def _safe_face_info(raw_image_name: str | None) -> dict[str, Any]:
    if not raw_image_name:
        return {}
    try:
        return parse_image_name(raw_image_name)
    except ValueError:
        return {}


def _build_tags(
    cube_faces: dict[str, Any],
    source_records: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    issue_config: dict[str, Any],
) -> list[dict[str, Any]]:
    tags = []
    for image_id, result in sorted(results.items()):
        source = source_records.get(image_id, {})
        raw_image_name = source.get("rawImageName") or source.get("raw_image_name")
        face_info = _safe_face_info(raw_image_name)
        pano = (cube_faces.get("panos") or {}).get(face_info.get("panoId", ""), {})
        boxes = {box.get("id"): box for box in _three_d_boxes(result) if isinstance(box, dict)}

        for task, instance in _task_instances(result):
            box = boxes.get(instance.get("3d_bbox_id")) or boxes.get(instance.get("2d_bbox_id"))
            if not isinstance(box, dict):
                continue
            anchor = _point(box.get("anchorPosition"))
            stem = _point(box.get("stemVector"))
            if not anchor or not stem:
                continue

            task_name = task.get("task_name") or ""
            tag = {
                "imageId": image_id,
                "rawImageName": raw_image_name,
                "resultKey": source.get("key") or source.get("resultKey"),
                "preprocessedKey": source.get("preprocessedKey"),
                "bbox2dId": instance.get("2d_bbox_id"),
                "bbox3dId": instance.get("3d_bbox_id"),
                "instanceId": instance.get("instance_id"),
                "taskId": task.get("task_id"),
                "taskName": task_name,
                "issueType": _issue_type(task, instance, issue_config),
                "class": instance.get("instance_class") or "",
                "trade": (instance.get("attributes") or {}).get("trade") or "",
                "reasoning": instance.get("instance_reasoning") or "",
                "objectType": _object_type(instance.get("instance_reasoning")),
                "panoId": box.get("panoId"),
                "roomId": (
                    pano.get("roomId")
                    or pano.get("room_id")
                    or (pano.get("room") or {}).get("id")
                    or ""
                ),
                "floorId": (
                    pano.get("floorId")
                    or pano.get("floor_id")
                    or (pano.get("room") or {}).get("floor_id")
                    or ""
                ),
                "locationId": pano.get("locationId") or pano.get("location_id") or "",
                "locationLabel": pano.get("room_name") or pano.get("room_label"),
                "faceIndex": face_info.get("faceIndex"),
                "faceDirection": face_info.get("faceDirection") or "",
                "rotation": _vector2d(box.get("rotation")),
                "anchorPosition": anchor,
                "stemVector": stem,
                "surfaceNormal": _point(box.get("surfaceNormal")),
                "surfaceNormalConfidence": box.get("surfaceNormalConfidence"),
                "surfacePlaneError": box.get("surfacePlaneError"),
                "surfaceNormalSamplePixelRadius": box.get("surfaceNormalSamplePixelRadius"),
            }
            tag["tagKey"] = _tag_key(tag)
            tags.append(tag)
    return tags


def _face_filter(nms_config: dict[str, Any]) -> set[str]:
    face_filter = nms_config.get("faceFilter")
    if not isinstance(face_filter, dict):
        return set()
    return {str(face).strip().lower() for face in face_filter.get("skipFaceDirections") or [] if str(face).strip()}


def _location_id(tag: dict[str, Any]) -> int:
    for value in (tag.get("bbox3dId"), tag.get("bbox2dId"), tag.get("instanceId")):
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    raise ValueError(f"NMS keeper is missing a numeric location id: {_tag_key(tag)}")


def _issue_location(tag: dict[str, Any]) -> dict[str, Any]:
    rotation = tag.get("rotation")
    if not isinstance(rotation, dict):
        raise ValueError(f"NMS tag is missing rotation for backend issue: {_tag_key(tag)}")

    return {
        "id": _location_id(tag),
        "panoId": str(tag.get("panoId") or ""),
        "label": str(tag.get("locationLabel") or ""),
        "rotation": rotation,
        "anchorPosition": tag["anchorPosition"],
        "stemVector": tag["stemVector"],
    }


def _unique_tags(tags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for tag in tags:
        key = _tag_key(tag)
        if key in seen:
            continue
        seen.add(key)
        result.append(tag)
    return result


def _highest_severity(tags: list[dict[str, Any]]) -> str:
    ranked = [tag for tag in tags if _severity_key(tag.get("class")) in SEVERITY_RANK]
    if not ranked:
        return str(tags[0].get("class") or "") if tags else ""
    highest = min(ranked, key=lambda tag: SEVERITY_RANK[_severity_key(tag.get("class"))])
    return str(highest.get("class") or "")


def _issue_info(
    keeper: dict[str, Any],
    *,
    members: list[dict[str, Any]],
    reasoning: str | None = None,
) -> IssueInfo:
    ordered_members = _unique_tags(
        [keeper, *[tag for tag in sorted(members, key=_tag_sort_key) if _tag_key(tag) != _tag_key(keeper)]]
    )
    images = [str(tag.get("imageId")) for tag in ordered_members if tag.get("imageId")]
    local_instance_ids = [tag.get("instanceId") for tag in ordered_members]

    return {
        "status": "",
        "created_by": "",
        "type": str(keeper.get("issueType") or "unknown"),
        "trade": str(keeper.get("trade") or ""),
        "severity": _highest_severity(ordered_members),
        "reasoning": str(reasoning or keeper.get("reasoning") or keeper.get("taskName") or ""),
        "fe_issue_id": "",
        "location": _issue_location(keeper),
        "images": images,
        "task_id": keeper.get("taskId"),
        "local_instance_ids": local_instance_ids,
    }


def _build_issue_reports(
    eligible_tags: list[dict[str, Any]],
    duplicate_keys: set[str],
    clusters: list[dict[str, Any]],
    project_name: str,
    issue_config: dict[str, Any],
    reasoning_overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    padding = int(issue_config.get("numberPadding", 4))
    counters: dict[tuple[str, str], int] = defaultdict(int)
    clusters_by_keeper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cluster in clusters:
        clusters_by_keeper[_tag_key(cluster["keeper"])].append(cluster)

    issue_reports = []
    for keeper in sorted(eligible_tags, key=_tag_sort_key):
        keeper_key = _tag_key(keeper)
        if keeper_key in duplicate_keys:
            continue

        raw_image_name = keeper.get("rawImageName") or keeper.get("imageId")
        issue_type = keeper.get("issueType") or "unknown"
        counter_key = (str(raw_image_name), str(issue_type))
        counters[counter_key] += 1
        issue_id = format_issue_id(project_name, raw_image_name, issue_type, counters[counter_key], padding)

        keeper_clusters = clusters_by_keeper.get(keeper_key, [])
        duplicate_members = [
            tag for cluster in keeper_clusters for tag in cluster.get("duplicates", []) if _tag_key(tag) in duplicate_keys
        ]
        final_reasoning = (reasoning_overrides or {}).get(keeper_key)

        issue_reports.append(
            {
                "issueId": issue_id,
                "passName": keeper_clusters[0].get("passName") if keeper_clusters else None,
                "group": keeper_clusters[0].get("group") if keeper_clusters else None,
                "threshold": keeper_clusters[0].get("threshold") if keeper_clusters else None,
                "keeper": _summarize(keeper, reasoning=final_reasoning),
                "issueInfo": _issue_info(
                    keeper,
                    members=duplicate_members,
                    reasoning=final_reasoning,
                ),
                "duplicateMembers": [_summarize(tag) for tag in sorted(duplicate_members, key=_tag_sort_key)],
            }
        )
    return issue_reports


def run_nms(
    nms_inputs: dict[str, dict[str, Any]],
    cube_faces: dict[str, Any],
    source_records: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    nms_config = {**DEFAULT_CONFIG, **(config.get("nms") or {})}
    issue_config = config.get("issueId") or {}
    project_name = str(config.get("projectName") or "project")

    all_tags = _build_tags(cube_faces, source_records, nms_inputs, issue_config)
    skip_faces = _face_filter(nms_config)
    eligible_tags = [
        tag for tag in all_tags if not skip_faces or str(tag.get("faceDirection", "")).lower() not in skip_faces
    ]
    eligible_keys = {_tag_key(tag) for tag in eligible_tags}
    skipped_by_filter = {_tag_key(tag) for tag in all_tags if _tag_key(tag) not in eligible_keys}

    duplicate_keys, clusters, pass_summaries = _run_all_passes(eligible_tags, _config_passes(nms_config))
    reasoning_overrides, nms_ai_summary = nms_ai_reasoning.process_issue_reasonings(
        eligible_tags,
        clusters,
        config.get("nmsAi") or {},
        duplicate_keys=duplicate_keys,
    )
    issue_reports = _build_issue_reports(
        eligible_tags,
        duplicate_keys,
        clusters,
        project_name,
        issue_config,
        reasoning_overrides=reasoning_overrides,
    )

    final_results = {image_id: result for image_id, result in sorted(nms_inputs.items())}
    summary = {
        "mode": "nms",
        "tagCount": len(all_tags),
        "eligibleTagCount": len(eligible_tags),
        "issueCount": len(issue_reports),
        "duplicateCount": len(duplicate_keys),
        "skippedByFaceFilterCount": len(skipped_by_filter),
        "clusterCount": len(clusters),
        "passes": pass_summaries,
        "nmsAi": nms_ai_summary,
        "issues": issue_reports,
    }
    return final_results, summary
