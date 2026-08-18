#!/usr/bin/env python3
"""Run production-equivalent QAQC NMS locally over existing nms_input.json files."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import boto3

from local_nms.control import disabled_remote_config, write_disabled_remote_config
from local_nms.runner import deep_merge, run_s3_nms
from s3_utils import fetch_master_project_id, master_config_key, read_json


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


def parse_s3_uri(uri: str) -> tuple[str, str, str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an S3 images URI, got: {uri}")
    prefix = parsed.path.strip("/").rstrip("/") + "/"
    parts = [part for part in prefix.split("/") if part]
    if len(parts) < 4 or parts[-3] != "use_cases" or parts[-1] != "images":
        raise ValueError("S3 URI must end with /use_cases/<use_case_id>/images/")
    project_id = "/".join(parts[:-3])
    use_case_id = parts[-2]
    if not project_id:
        raise ValueError("S3 URI is missing its project id.")
    return parsed.netloc, prefix, project_id, use_case_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images_uri", help="S3 use-case images prefix ending in /images/")
    parser.add_argument("--profile", help="Optional AWS profile; otherwise boto3 uses the environment")
    parser.add_argument("--config", type=Path, help="Optional JSON overrides merged onto remote qaqc_config.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--no-s3-report",
        action="store_true",
        help="Do not write qaqc/local_nms/nms_report.json and finalization.json to S3.",
    )
    parser.add_argument(
        "--publish-issues",
        action="store_true",
        help="Publish issue payloads directly from the laptop; requires QAQC API environment variables.",
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
    bucket, images_prefix, project_id, use_case_id = parse_s3_uri(args.images_uri)
    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    s3_client = session.client("s3")

    master_project_id = fetch_master_project_id(s3_client, bucket, use_case_id)
    config_key = master_config_key(master_project_id, use_case_id)
    remote_config = read_json(s3_client, bucket, config_key)
    if remote_config.get("processingMode") != "perImage":
        raise RuntimeError("Remote qaqc_config.json must have processingMode=perImage before using this helper.")
    disabled_config = disabled_remote_config(remote_config)
    write_disabled_remote_config(s3_client, bucket, config_key, disabled_config)
    if args.config:
        override = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(override, dict):
            raise ValueError(f"Config override must contain a JSON object: {args.config}")
        remote_config = deep_merge(remote_config, override)

    try:
        use_case_prefix = images_prefix.rstrip("/").rsplit("/images", 1)[0]
        cube_faces = read_json(s3_client, bucket, f"{use_case_prefix}/cube_faces.json")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result = run_s3_nms(
            s3_client,
            bucket=bucket,
            project_id=project_id,
            use_case_id=use_case_id,
            images_prefix=images_prefix,
            cube_faces=cube_faces,
            remote_config=remote_config,
            run_id=run_id,
            output_root=args.output_root,
            write_s3_report=not args.no_s3_report,
            publish_issues=args.publish_issues,
        )
        logging.info("NMS report: %s", result.output_dir / "nms_report.json")
        if result.report_key:
            logging.info("S3 report: s3://%s/%s", bucket, result.report_key)
    finally:
        write_disabled_remote_config(s3_client, bucket, config_key, disabled_config)
        logging.info("Remote QAQC left disabled; local NMS owns this workflow.")


if __name__ == "__main__":
    main()
