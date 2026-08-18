from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_nms import engine
from local_nms.ai import reasoning as nms_ai_reasoning
from local_nms.publisher import per_image_finalization_id, publish_issue_report, write_issue_files
from s3_utils import list_object_keys, put_json, read_json


DEFAULT_ISSUE_CONFIG = {"numberPadding": 4, "issueTypeSource": "task_code"}


@dataclass(frozen=True)
class LocalNmsRun:
    report: dict[str, Any]
    marker: dict[str, Any]
    output_dir: Path
    report_key: str | None


def isoformat_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge config objects the same way as the remote pipeline."""
    output = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = deep_merge(output[key], value)
        else:
            output[key] = deepcopy(value)
    return output


def effective_config(remote_config: dict[str, Any], project_id: str) -> dict[str, Any]:
    config = deepcopy(remote_config)
    config["projectName"] = config.get("projectName") or project_id
    config["issueId"] = deep_merge(DEFAULT_ISSUE_CONFIG, config.get("issueId") or {})
    config["nms"] = deep_merge(engine.DEFAULT_CONFIG, config.get("nms") or {})
    config["nms"]["enabled"] = True
    config["nmsAi"] = deep_merge(nms_ai_reasoning.DEFAULT_CONFIG, config.get("nmsAi") or {})
    return config


def load_nms_context(
    s3_client,
    bucket: str,
    images_prefix: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    keys = [
        key
        for key in list_object_keys(s3_client, bucket, images_prefix)
        if key.endswith("/nms_input.json")
    ]
    for key in keys:
        artifact = read_json(s3_client, bucket, key)
        if artifact.get("status") == "skipped":
            continue
        image_id = str(artifact.get("imageId") or key.rsplit("/", 2)[-2])
        result = artifact.get("result")
        source = artifact.get("source")
        if not isinstance(result, dict) or not isinstance(source, dict):
            logging.warning("Ignoring malformed local NMS input: s3://%s/%s", bucket, key)
            continue
        results[image_id] = result
        sources[image_id] = source
    return results, sources


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_s3_nms(
    s3_client,
    *,
    bucket: str,
    project_id: str,
    use_case_id: str,
    images_prefix: str,
    cube_faces: dict[str, Any],
    remote_config: dict[str, Any],
    run_id: str,
    output_root: Path,
    write_s3_report: bool = True,
    publish_issues: bool = False,
) -> LocalNmsRun:
    """Run production-equivalent NMS locally over the 3D-enriched S3 inputs."""
    results, sources = load_nms_context(s3_client, bucket, images_prefix)
    if not results:
        raise RuntimeError(f"No ready nms_input.json files found under s3://{bucket}/{images_prefix}")

    config = effective_config(remote_config, project_id)
    final_results, report = engine.run_nms(results, cube_faces, sources, config)
    if set(final_results) != set(results):
        raise RuntimeError("Local NMS returned a different image-id set than its inputs.")

    output_dir = output_root / project_id / use_case_id / run_id
    _write_json(output_dir / "nms_report.json", report)
    issue_files_written = write_issue_files(report, output_dir)
    finalization_id = per_image_finalization_id(project_id, use_case_id)
    backend_issues_created = 0
    receipt_path = output_dir / "publish_receipts.json"
    if publish_issues:
        backend_issues_created = publish_issue_report(
            report,
            project_id=project_id,
            use_case_id=use_case_id,
            finalization_id=finalization_id,
            receipts_path=receipt_path,
        )

    marker = {
        "schemaVersion": 1,
        "status": "finalized",
        "mode": "local-nms",
        "finalizationId": finalization_id,
        "runId": run_id,
        "completedAt": isoformat_utc(),
        "inputCount": len(results),
        "nms": {
            "tagCount": report.get("tagCount", 0),
            "issueCount": report.get("issueCount", 0),
            "duplicateCount": report.get("duplicateCount", 0),
            "skippedByFaceFilterCount": report.get("skippedByFaceFilterCount", 0),
            "clusterCount": report.get("clusterCount", 0),
        },
        "issueFiles": {"written": issue_files_written},
        "backendIssues": {
            "created": backend_issues_created,
            "requested": publish_issues,
            "published": publish_issues and backend_issues_created == report.get("issueCount", 0),
            "receiptFile": receipt_path.name if publish_issues else None,
        },
    }
    _write_json(output_dir / "finalization.json", marker)

    report_key = None
    if write_s3_report:
        qaqc_prefix = images_prefix.rstrip("/").rsplit("/images", 1)[0] + "/qaqc/local_nms"
        report_key = f"{qaqc_prefix}/nms_report.json"
        put_json(
            s3_client,
            bucket,
            report_key,
            {
                "schemaVersion": 1,
                "status": "ready",
                "finalizationId": finalization_id,
                "runId": run_id,
                "completedAt": marker["completedAt"],
                "report": report,
            },
        )
        put_json(s3_client, bucket, f"{qaqc_prefix}/finalization.json", marker)

    logging.info(
        "Local NMS finalized: inputs=%s tags=%s issues=%s duplicates=%s output=%s",
        len(results),
        report.get("tagCount", 0),
        report.get("issueCount", 0),
        report.get("duplicateCount", 0),
        output_dir,
    )
    return LocalNmsRun(report=report, marker=marker, output_dir=output_dir, report_key=report_key)
