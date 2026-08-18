from __future__ import annotations

import math
from typing import Any

from local_nms.ai import client, output_models, prompt_builder


DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "model": dict(client.DEFAULT_MODEL),
    "minimumClusterSize": 3,
    "maxMembersPerCluster": 5,
    "maxReasoningsPerCall": 100,
    "timeoutSeconds": 60,
    "maxCompletionTokens": client.DEFAULT_MAX_COMPLETION_TOKENS,
    "prompts": {},
}


def _tag_key(tag: dict[str, Any]) -> str:
    return str(tag.get("tagKey") or "")


def _cluster_members(cluster: dict[str, Any]) -> list[dict[str, Any]]:
    members = [cluster.get("keeper"), *(cluster.get("duplicates") or [])]
    unique_members: dict[str, dict[str, Any]] = {}
    for member in members:
        if not isinstance(member, dict):
            continue
        key = _tag_key(member)
        if key:
            unique_members.setdefault(key, member)
    return list(unique_members.values())


def _members_by_keeper(clusters: list[dict[str, Any]], duplicate_keys: set[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for cluster in clusters:
        keeper = cluster.get("keeper")
        if not isinstance(keeper, dict):
            continue
        keeper_key = _tag_key(keeper)
        if not keeper_key or keeper_key in duplicate_keys:
            continue
        grouped.setdefault(keeper_key, {})
        for member in _cluster_members(cluster):
            grouped[keeper_key].setdefault(_tag_key(member), member)
    return {keeper_key: list(members.values()) for keeper_key, members in grouped.items() if len(members) > 1}


def _centroid_sample(members: list[dict[str, Any]], keeper_key: str, limit: int) -> list[dict[str, Any]]:
    if len(members) <= limit:
        return members

    centroid = tuple(
        sum(float(member["anchorPosition"][axis]) for member in members) / len(members)
        for axis in ("x", "y", "z")
    )

    def sort_key(member: dict[str, Any]) -> tuple[float, str]:
        anchor = member["anchorPosition"]
        distance = math.dist((float(anchor["x"]), float(anchor["y"]), float(anchor["z"])), centroid)
        return distance, _tag_key(member)

    selected = sorted(members, key=sort_key)[:limit]
    keeper = next(member for member in members if _tag_key(member) == keeper_key)
    if all(_tag_key(member) != keeper_key for member in selected):
        selected[-1] = keeper
        selected.sort(key=sort_key)
    return selected


def _configured_prompts(settings: dict[str, Any]) -> dict[str, str]:
    prompts = settings.get("prompts")
    if not isinstance(prompts, dict):
        raise ValueError("nmsAi.prompts must be an object.")
    configured = {}
    for key in ("system", "merge", "cleanup"):
        value = prompts.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"nmsAi.prompts.{key} must be a non-empty string.")
        configured[key] = value.strip()
    return configured


def _reasoning_batches(items: list[dict[str, Any]], limit: int) -> list[list[dict[str, Any]]]:
    batches = []
    current_batch = []
    current_reasoning_count = 0
    for item in items:
        reasoning_count = len(item["members"])
        if current_batch and current_reasoning_count + reasoning_count > limit:
            batches.append(current_batch)
            current_batch = []
            current_reasoning_count = 0
        current_batch.append(item)
        current_reasoning_count += reasoning_count
    if current_batch:
        batches.append(current_batch)
    return batches


def process_issue_reasonings(
    eligible_tags: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    config: dict[str, Any] | None,
    *,
    duplicate_keys: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    settings = {**DEFAULT_CONFIG, **(config or {})}
    if not settings.get("enabled"):
        return {}, {"enabled": False}

    minimum_cluster_size = int(settings["minimumClusterSize"])
    max_members_per_cluster = int(settings["maxMembersPerCluster"])
    max_reasonings_per_call = int(settings["maxReasoningsPerCall"])
    if minimum_cluster_size < 2:
        raise ValueError("nmsAi.minimumClusterSize must be at least 2.")
    if max_members_per_cluster < minimum_cluster_size:
        raise ValueError("nmsAi.maxMembersPerCluster must be at least minimumClusterSize.")
    if max_reasonings_per_call < max_members_per_cluster:
        raise ValueError("nmsAi.maxReasoningsPerCall must be at least maxMembersPerCluster.")
    timeout = int(settings["timeoutSeconds"])
    max_completion_tokens = int(settings["maxCompletionTokens"])
    if timeout < 1:
        raise ValueError("nmsAi.timeoutSeconds must be greater than 0.")
    if max_completion_tokens < 1:
        raise ValueError("nmsAi.maxCompletionTokens must be greater than 0.")
    prompts = _configured_prompts(settings)

    resolved_duplicate_keys = duplicate_keys or set()
    grouped = _members_by_keeper(clusters, resolved_duplicate_keys)
    keepers = [
        tag for tag in sorted(eligible_tags, key=_tag_key)
        if _tag_key(tag) and _tag_key(tag) not in resolved_duplicate_keys
    ]
    items = []
    sampled_cluster_count = 0
    merge_issue_count = 0

    for index, keeper in enumerate(keepers, start=1):
        keeper_key = _tag_key(keeper)
        members = grouped.get(keeper_key, [keeper])
        operation = "merge" if len(members) >= minimum_cluster_size else "cleanup"
        if operation == "merge":
            input_members = _centroid_sample(members, keeper_key, max_members_per_cluster)
            sampled_cluster_count += len(input_members) < len(members)
            merge_issue_count += 1
        else:
            input_members = [keeper]
        issue_id = f"issue_{index:04d}"
        items.append(
            {
                "issue_id": issue_id,
                "operation": operation,
                "keeper_key": keeper_key,
                "members": input_members,
            }
        )

    if not items:
        return {}, {
            "enabled": True,
            "aiCallCount": 0,
            "processedIssueCount": 0,
            "cleanupIssueCount": 0,
            "mergedIssueCount": 0,
            "sampledClusterCount": 0,
        }

    batches = _reasoning_batches(items, max_reasonings_per_call)
    results_by_issue_id = {}
    for call_items in batches:
        raw = client.process_reasonings(
            prompts["system"],
            prompt_builder.build_user_prompt(call_items, prompts),
            model=settings.get("model"),
            timeout=timeout,
            max_completion_tokens=max_completion_tokens,
        )
        output = output_models.normalize_reasoning_response(raw)
        call_results = {}
        for item in output.items:
            if item.issue_id in call_results:
                raise ValueError(f"NMS AI reasoning response repeated issue_id={item.issue_id!r}.")
            call_results[item.issue_id] = item.reasoning
        expected_issue_ids = {item["issue_id"] for item in call_items}
        returned_issue_ids = set(call_results)
        if returned_issue_ids != expected_issue_ids:
            missing = sorted(expected_issue_ids - returned_issue_ids)
            unknown = sorted(returned_issue_ids - expected_issue_ids)
            raise ValueError(f"NMS AI reasoning response IDs do not match input: missing={missing}, unknown={unknown}.")
        results_by_issue_id.update(call_results)

    overrides = {
        item["keeper_key"]: results_by_issue_id[item["issue_id"]]
        for item in items
    }
    return overrides, {
        "enabled": True,
        "aiCallCount": len(batches),
        "processedIssueCount": len(items),
        "cleanupIssueCount": len(items) - merge_issue_count,
        "mergedIssueCount": merge_issue_count,
        "sampledClusterCount": sampled_cluster_count,
    }
