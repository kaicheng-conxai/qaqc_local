#!/usr/bin/env python3
"""Publish backend issues from an existing local NMS report without rerunning NMS."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from local_nms.publisher import (
    per_image_finalization_id,
    publish_issue_report,
    validate_issue_report,
    validate_publish_environment,
)


def load_report(path: Path) -> dict:
    report_path = path / "nms_report.json" if path.is_dir() else path
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"NMS report must contain a JSON object: {report_path}")
    report = payload.get("report") if isinstance(payload.get("report"), dict) else payload
    if not isinstance(report.get("issues"), list):
        raise ValueError(f"NMS report is missing its issues list: {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="nms_report.json or its local run directory")
    parser.add_argument("--project-id", required=True, help="Root project id used by the issue API")
    parser.add_argument("--use-case-id", required=True, help="Use-case id used by the issue API")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the report and API environment without making HTTP requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    report = load_report(args.report)
    issue_count = validate_issue_report(report)
    settings = validate_publish_environment(args.project_id, args.use_case_id)
    if args.dry_run:
        logging.info(
            "Dry run passed: issues=%s endpoint=%s apiKeyHeader=%s; no HTTP requests sent",
            issue_count,
            settings["endpoint"],
            settings["apiKeyHeader"],
        )
        return
    created = publish_issue_report(
        report,
        project_id=args.project_id,
        use_case_id=args.use_case_id,
        finalization_id=per_image_finalization_id(args.project_id, args.use_case_id),
        receipts_path=args.report.parent / "publish_receipts.json",
    )
    logging.info("Published %s backend issue(s) from %s", created, args.report)


if __name__ == "__main__":
    main()
