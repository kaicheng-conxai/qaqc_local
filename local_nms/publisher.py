from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ISSUE_ROUTE = "/internal/qaqc/projects/{project_id}/use-cases/{use_case_id}/issues"
DEFAULT_API_KEY_HEADER = "X-API-Key-s2s"
SUPPORTED_API_KEY_HEADERS = {DEFAULT_API_KEY_HEADER, "X-API-Key"}
DEFAULT_API_BASE_URL = "http://127.0.0.1:18000"


def per_image_finalization_id(project_id: str, use_case_id: str) -> str:
    """Return the same stable finalization id used by remote per-image NMS."""
    value = f"{project_id}/{use_case_id}/per-image-nms"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def issue_payload(issue_info: dict[str, Any]) -> dict[str, Any]:
    """Build the exact request body used by the remote QAQC issue writer."""
    payload = dict(issue_info)
    payload.pop("fe_issue_id", None)
    payload["location"] = [issue_info["location"]]
    return payload


def validate_publish_environment(project_id: str, use_case_id: str) -> dict[str, str]:
    """Validate local issue API settings without sending a request."""
    api_key = os.getenv("QAQC_INTERNAL_API_KEY")
    if not api_key:
        raise ValueError("QAQC_INTERNAL_API_KEY is required to publish backend issues locally.")
    api_key_header = os.getenv("QAQC_API_KEY_HEADER", DEFAULT_API_KEY_HEADER)
    if api_key_header not in SUPPORTED_API_KEY_HEADERS:
        raise ValueError("QAQC_API_KEY_HEADER must be X-API-Key-s2s or X-API-Key.")
    return {
        "endpoint": _issue_endpoint(project_id, use_case_id),
        "apiKeyHeader": api_key_header,
    }


def validate_issue_report(nms_report: dict[str, Any]) -> int:
    """Validate every issue payload locally and return the publishable count."""
    issue_count = 0
    for issue in nms_report.get("issues") or []:
        issue_info = issue.get("issueInfo") if isinstance(issue, dict) else None
        if not isinstance(issue_info, dict):
            raise ValueError(f"NMS report issue at index {issue_count} is missing issueInfo.")
        issue_payload(issue_info)
        issue_count += 1
    return issue_count


def write_issue_files(nms_report: dict[str, Any], output_dir: Path) -> int:
    issues_dir = output_dir / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for issue in nms_report.get("issues") or []:
        issue_info = issue.get("issueInfo") if isinstance(issue, dict) else None
        if not isinstance(issue_info, dict):
            continue
        written += 1
        target = issues_dir / f"issue_{written:04d}.json"
        target.write_text(
            json.dumps(issue_payload(issue_info), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return written


def _issue_endpoint(project_id: str, use_case_id: str) -> str:
    base_url = os.getenv("QAQC_API_BASE_URL", DEFAULT_API_BASE_URL).strip()
    if not base_url:
        raise ValueError("QAQC_API_BASE_URL must not be empty.")
    route = ISSUE_ROUTE.format(project_id=project_id, use_case_id=use_case_id)
    return base_url.rstrip("/") + route


def _create_issue(
    issue_info: dict[str, Any],
    *,
    project_id: str,
    use_case_id: str,
    idempotency_key: str,
) -> dict[str, Any]:
    settings = validate_publish_environment(project_id, use_case_id)
    api_key = os.environ["QAQC_INTERNAL_API_KEY"]

    request = Request(
        settings["endpoint"],
        data=json.dumps(issue_payload(issue_info), ensure_ascii=False).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            settings["apiKeyHeader"]: api_key,
            "Idempotency-Key": idempotency_key,
        },
        method="POST",
    )
    try:
        with urlopen(request) as response:
            status = response.getcode()
            response_body = response.read().decode("utf-8", errors="replace")
            if status < 200 or status >= 300:
                raise RuntimeError(f"Issue API returned HTTP {status}.")
            logging.info("Issue API accepted status=%s", status)
            return {
                "status": status,
                "body": _response_body(response_body),
            }
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Issue API returned HTTP {error.code}: {body[:500]}") from error
    except URLError as error:
        raise RuntimeError(f"Issue API request failed: {error.reason}") from error


def _response_body(value: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value[:2000]


def publish_issue_report(
    nms_report: dict[str, Any],
    *,
    project_id: str,
    use_case_id: str,
    finalization_id: str,
    receipts_path: Path | None = None,
) -> int:
    """Publish local NMS issues with remote-compatible payloads and idempotency keys.

    A receipt file is written after each successful request when ``receipts_path`` is
    supplied. This makes a run auditable even if a later issue fails to publish.
    """
    created = 0
    receipts: list[dict[str, Any]] = []
    settings = validate_publish_environment(project_id, use_case_id) if receipts_path else None
    if receipts_path:
        receipts_path.parent.mkdir(parents=True, exist_ok=True)
        receipts_path.write_text(
            json.dumps(
                {
                    "endpoint": settings["endpoint"],
                    "apiKeyHeader": settings["apiKeyHeader"],
                    "finalizationId": finalization_id,
                    "publishedCount": 0,
                    "receipts": [],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    for issue in nms_report.get("issues") or []:
        issue_info = issue.get("issueInfo") if isinstance(issue, dict) else None
        if not isinstance(issue_info, dict):
            continue
        issue_id = str(issue.get("issueId") or "")
        if not issue_id:
            issue_id = hashlib.sha256(
                json.dumps(issue_info, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest()
        idempotency_key = hashlib.sha256(f"{finalization_id}:{issue_id}".encode("utf-8")).hexdigest()
        response = _create_issue(
            issue_info,
            project_id=project_id,
            use_case_id=use_case_id,
            idempotency_key=idempotency_key,
        )
        created += 1
        if receipts_path:
            receipts.append(
                {
                    "issueId": issue_id,
                    "idempotencyKey": idempotency_key,
                    "endpoint": settings["endpoint"],
                    **response,
                }
            )
            receipts_path.write_text(
                json.dumps(
                    {
                        "endpoint": settings["endpoint"],
                        "apiKeyHeader": settings["apiKeyHeader"],
                        "finalizationId": finalization_id,
                        "publishedCount": created,
                        "receipts": receipts,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    return created
