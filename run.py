#!/usr/bin/env python3
"""Run Matterport 3D backfill and production-equivalent NMS locally.

The remote QAQC service stays disabled. Issue request payloads are written locally;
publishing them to the backend is an explicit opt-in.

Example:
    python3 tmp/local_3d_backfill_issues/run.py \
      s3://dev-sitelens-var/<project>/use_cases/<use_case>/images/

Use --force to regenerate every existing nms_input.json for the submitted
results and rerun local NMS.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3

import anchors as anchors3d
from artifacts import build_nms_input
from cube_faces import expected_image_names, image_index
from local_nms.control import disabled_remote_config, write_disabled_remote_config
from local_nms.runner import run_s3_nms
from matterport import close_persistent_sessions
from s3_utils import (
    content_disposition_filename,
    fetch_master_project_id,
    head_object,
    list_object_keys,
    master_config_key,
    put_json,
    put_json_if_absent,
    read_json,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_NMS_OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    result_key: str
    preprocessed_key: str
    nms_input_key: str
    raw_image_name: str


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected an S3 URI such as s3://bucket/project/use_cases/use_case/images/: {uri}")
    return parsed.netloc, parsed.path.strip("/").rstrip("/") + "/"


def parse_images_prefix(prefix: str) -> tuple[str, str]:
    parts = [part for part in prefix.strip("/").split("/") if part]
    try:
        use_cases_index = parts.index("use_cases")
    except ValueError as error:
        raise ValueError(f"S3 prefix must contain /use_cases/<use_case>/images/: {prefix}") from error
    if len(parts) != use_cases_index + 3 or parts[-1] != "images":
        raise ValueError(f"Pass the use-case images prefix ending in /images/: {prefix}")
    use_case_prefix = "/".join(parts[: use_cases_index + 2])
    return use_case_prefix, prefix


def load_local_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Local config must contain a JSON object: {path}")
    workers = max(1, int(config.get("workers", 4)))
    return {
        **config,
        "workers": workers,
        "chunkSize": max(1, int(config.get("chunkSize", 1))),
        "settleMs": max(0, int(config.get("settleMs", 120))),
        "persistentSession": bool(config.get("persistentSession", True)),
        "headed": bool(config.get("headed", True)),
        "sessionIdleSeconds": max(0.0, float(config.get("sessionIdleSeconds", 300))),
    }


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def isoformat_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def image_id_from_key(result_key: str) -> str:
    return result_key.rstrip("/").split("/")[-2]


def direct_child_keys(s3_client, bucket: str, images_prefix: str, filename: str) -> list[str]:
    prefix = images_prefix.rstrip("/") + "/"
    keys = []
    for key in list_object_keys(s3_client, bucket, prefix):
        relative = key[len(prefix) :]
        if relative.count("/") == 1 and relative.endswith(f"/{filename}"):
            keys.append(key)
    return keys


def raw_image_name_for_result(s3_client, bucket: str, result_key: str) -> str | None:
    preprocessed_key = result_key.rsplit("/", 1)[0] + "/preprocessed.jpg"
    metadata = head_object(s3_client, bucket, preprocessed_key)
    if not metadata:
        return None
    return content_disposition_filename(metadata.content_disposition)


def find_result_records(
    s3_client,
    bucket: str,
    images_prefix: str,
    expected_names: set[str],
) -> tuple[dict[str, ImageRecord], list[str], list[str]]:
    records: dict[str, ImageRecord] = {}
    duplicates: list[str] = []
    unexpected: list[str] = []
    for result_key in direct_child_keys(s3_client, bucket, images_prefix, "result.json"):
        raw_name = raw_image_name_for_result(s3_client, bucket, result_key)
        if not raw_name:
            logging.warning("Skipping result without preprocessed.jpg filename: s3://%s/%s", bucket, result_key)
            continue
        if raw_name not in expected_names:
            unexpected.append(raw_name)
            continue
        if raw_name in records:
            duplicates.append(raw_name)
            continue
        image_prefix = result_key.rsplit("/", 1)[0]
        records[raw_name] = ImageRecord(
            image_id=image_id_from_key(result_key),
            result_key=result_key,
            preprocessed_key=f"{image_prefix}/preprocessed.jpg",
            nms_input_key=f"{image_prefix}/nms_input.json",
            raw_image_name=raw_name,
        )
    return records, duplicates, unexpected


def existing_nms_names(
    s3_client,
    bucket: str,
    images_prefix: str,
    expected_names: set[str],
) -> tuple[set[str], list[str], int]:
    names: set[str] = set()
    unreadable: list[str] = []
    tag_count = 0
    for key in direct_child_keys(s3_client, bucket, images_prefix, "nms_input.json"):
        try:
            artifact = read_json(s3_client, bucket, key)
        except Exception as error:
            unreadable.append(f"{key}: {error}")
            continue
        source = artifact.get("source") if isinstance(artifact, dict) else None
        raw_name = source.get("rawImageName") if isinstance(source, dict) else None
        if isinstance(raw_name, str) and raw_name in expected_names:
            names.add(raw_name)
        result = artifact.get("result") if isinstance(artifact, dict) else None
        prediction = result.get("prediction") if isinstance(result, dict) else None
        tasks = prediction.get("tasks") if isinstance(prediction, dict) else None
        if isinstance(tasks, list):
            tag_count += sum(
                len(task.get("task_result") or [])
                for task in tasks
                if isinstance(task, dict) and isinstance(task.get("task_result"), list)
            )
    return names, unreadable, tag_count


def completion_target_names(
    cube_face_names: set[str],
    result_names: set[str],
    *,
    require_all_cube_faces: bool,
) -> set[str]:
    """Choose the submitted image set, or the full model manifest in strict mode."""
    return set(cube_face_names if require_all_cube_faces else result_names)


def log_summary(
    *,
    expected_count: int,
    result_count: int,
    existing_before_count: int,
    local_candidate_count: int,
    final_input_count: int,
    tag_count: int,
    failure_count: int,
    force: bool,
    started_at: float,
    finalization: dict[str, Any] | None = None,
) -> None:
    nms = finalization.get("nms") if isinstance(finalization, dict) else {}
    nms = nms if isinstance(nms, dict) else {}
    logging.info(
        "SUMMARY images_expected=%s results=%s nms_inputs=%s local_processed=%s "
        "skipped_existing=%s input_tags=%s nms_tags=%s nms_issues=%s failures=%s "
        "elapsed_seconds=%.2f status=%s",
        expected_count,
        result_count,
        final_input_count,
        max(0, local_candidate_count - failure_count),
        0 if force else existing_before_count,
        tag_count,
        nms.get("tagCount", "n/a"),
        nms.get("issueCount", "n/a"),
        failure_count,
        time.monotonic() - started_at,
        (finalization or {}).get("status", "local_backfill_failed"),
    )


def build_source(s3_client, bucket: str, record: ImageRecord) -> dict[str, Any]:
    result_head = head_object(s3_client, bucket, record.result_key)
    return {
        "resultKey": record.result_key,
        "preprocessedKey": record.preprocessed_key,
        "resultVersionId": result_head.version_id if result_head else None,
        "rawImageName": record.raw_image_name,
    }


def process_partition(
    worker_index: int,
    records: list[ImageRecord],
    s3_client,
    bucket: str,
    cube_faces: dict[str, Any],
    local_config: dict[str, Any],
    session_scope: str,
    run_id: str,
    force: bool,
) -> list[str]:
    model_id = str(cube_faces.get("modelId") or cube_faces.get("model_id") or "")
    worker_scope = f"{session_scope}:worker-{worker_index}"
    three_d_config = {
        **local_config,
        "workers": 1,
        "chunkSize": local_config["chunkSize"],
        "sessionScope": worker_scope,
    }
    failures: list[str] = []
    try:
        for record in records:
            try:
                if not force and head_object(s3_client, bucket, record.nms_input_key):
                    logging.info("Skipping existing nms_input.json: s3://%s/%s", bucket, record.nms_input_key)
                    continue

                result = read_json(s3_client, bucket, record.result_key)
                if not isinstance(result, dict):
                    raise ValueError("result.json must contain a JSON object")
                source = build_source(s3_client, bucket, record)
                backfilled, anchor_metadata = anchors3d.backfill_with_metadata(
                    {record.image_id: result},
                    cube_faces,
                    {record.image_id: record.raw_image_name},
                    three_d_config,
                )
                artifact = build_nms_input(
                    record.image_id,
                    source,
                    backfilled[record.image_id],
                    anchor_metadata.get(record.image_id) or {},
                    run_id,
                    isoformat_utc(),
                )
                artifact["status"] = "ready"
                if force:
                    put_json(s3_client, bucket, record.nms_input_key, artifact)
                elif not put_json_if_absent(s3_client, bucket, record.nms_input_key, artifact):
                    logging.info("Another process created nms_input.json: s3://%s/%s", bucket, record.nms_input_key)
                else:
                    logging.info("Wrote nms_input.json: s3://%s/%s", bucket, record.nms_input_key)
            except Exception as error:
                failures.append(f"{record.raw_image_name}: {error}")
                logging.exception("3D backfill failed for %s", record.raw_image_name)
    finally:
        if model_id and local_config["persistentSession"]:
            close_persistent_sessions(worker_scope, model_id)
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images_uri", help="S3 use-case images prefix ending in /images/")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Standalone local config JSON")
    parser.add_argument("--force", action="store_true", help="Regenerate every existing nms_input.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headed", dest="headed", action="store_true", help="Show the Matterport browser windows")
    mode.add_argument("--headless", dest="headed", action="store_false", help="Run Matterport without visible windows")
    parser.set_defaults(headed=None)
    parser.add_argument("--profile", help="Optional AWS profile; otherwise boto3 uses the environment")
    parser.add_argument("--nms-output-root", type=Path, default=DEFAULT_NMS_OUTPUT_ROOT)
    parser.add_argument("--no-s3-report", action="store_true", help="Keep local NMS output on disk only")
    parser.add_argument(
        "--require-all-cube-faces",
        action="store_true",
        help="Require result.json and nms_input.json for every face in the full cube_faces model.",
    )
    parser.add_argument(
        "--publish-issues",
        action="store_true",
        help="Publish final issue payloads directly; requires QAQC API environment variables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.monotonic()
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    bucket, images_prefix = parse_s3_uri(args.images_uri)
    use_case_prefix, images_prefix = parse_images_prefix(images_prefix)
    project_id = use_case_prefix.split("/use_cases/", 1)[0]
    use_case_id = use_case_prefix.split("/use_cases/", 1)[1]
    use_case_id = use_case_id.split("/", 1)[0]
    local_config = load_local_config(args.config)
    if args.headed is not None:
        local_config["headed"] = args.headed
    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    s3_client = session.client("s3")

    master_project_id = fetch_master_project_id(s3_client, bucket, use_case_id)
    config_key = master_config_key(master_project_id, use_case_id)
    remote_config = read_json(s3_client, bucket, config_key)
    if remote_config.get("processingMode") != "perImage":
        raise RuntimeError("Remote qaqc_config.json must have processingMode=perImage before using this helper.")

    cube_faces_key = f"{use_case_prefix}/cube_faces.json"
    cube_faces = read_json(s3_client, bucket, cube_faces_key)
    expected_names = expected_image_names(image_index(cube_faces))
    result_records, duplicates, unexpected = find_result_records(
        s3_client, bucket, images_prefix, expected_names
    )
    existing_names, unreadable, _initial_tag_count = existing_nms_names(
        s3_client, bucket, images_prefix, expected_names
    )
    result_names = set(result_records)
    target_names = completion_target_names(
        expected_names,
        result_names,
        require_all_cube_faces=args.require_all_cube_faces,
    )
    existing_before_count = len(existing_names & target_names)
    logging.info(
        "Cube-face manifest=%s submitted results=%s target images=%s existing target inputs=%s "
        "manifest faces without results=%s",
        len(expected_names),
        len(result_records),
        len(target_names),
        existing_before_count,
        len(expected_names - result_names),
    )
    if duplicates:
        raise RuntimeError(f"Duplicate result.json files map to the same raw image: {sorted(set(duplicates))}")
    if unexpected:
        logging.warning("Ignoring result.json files not listed in cube_faces.json: %s", sorted(set(unexpected)))
    if unreadable:
        logging.warning("Ignoring unreadable existing nms_input.json files: %s", unreadable)
    if not result_records:
        raise RuntimeError(f"No supported result.json files found under s3://{bucket}/{images_prefix}")

    run_id = utc_run_id()
    session_scope = f"{bucket}:{use_case_prefix}:local-3d-backfill:{run_id}"
    local_candidate_count = 0
    failure_count = 0
    disabled_config = disabled_remote_config(remote_config)
    write_disabled_remote_config(s3_client, bucket, config_key, disabled_config)
    try:
        pending = [
            record
            for raw_name, record in sorted(result_records.items())
            if args.force or raw_name not in existing_names
        ]
        local_candidate_count = len(pending)
        partitions = [pending[index::local_config["workers"]] for index in range(local_config["workers"])]
        partitions = [partition for partition in partitions if partition]
        failures: list[str] = []
        if partitions:
            with ThreadPoolExecutor(max_workers=min(local_config["workers"], len(partitions))) as executor:
                futures = [
                    executor.submit(
                        process_partition,
                        index,
                        partition,
                        s3_client,
                        bucket,
                        cube_faces,
                        local_config,
                        session_scope,
                        run_id,
                        args.force,
                    )
                    for index, partition in enumerate(partitions)
                ]
                for future in as_completed(futures):
                    failures.extend(future.result())
        failure_count = len(failures)
        if failures:
            final_names, _final_unreadable, tag_count = existing_nms_names(
                s3_client, bucket, images_prefix, expected_names
            )
            log_summary(
                expected_count=len(target_names),
                result_count=len(result_records),
                existing_before_count=existing_before_count,
                local_candidate_count=local_candidate_count,
                final_input_count=len(final_names),
                tag_count=tag_count,
                failure_count=failure_count,
                force=args.force,
                started_at=started_at,
            )
            raise RuntimeError(f"3D backfill failed for {len(failures)} image(s): {failures[:5]}")

        existing_names, unreadable, tag_count = existing_nms_names(
            s3_client, bucket, images_prefix, expected_names
        )
        final_result_records, final_duplicates, final_unexpected = find_result_records(
            s3_client, bucket, images_prefix, expected_names
        )
        if final_duplicates:
            raise RuntimeError(
                "Duplicate result.json files map to the same raw image: "
                f"{sorted(set(final_duplicates))}"
            )
        if final_unexpected:
            logging.warning(
                "Ignoring result.json files not listed in cube_faces.json: %s",
                sorted(set(final_unexpected)),
            )
        final_result_names = set(final_result_records)
        new_result_names = final_result_names - result_names
        if new_result_names:
            raise RuntimeError(
                "New result.json files arrived during local 3D processing; rerun so they are included: "
                f"{sorted(new_result_names)}"
            )
        target_names = completion_target_names(
            expected_names,
            final_result_names,
            require_all_cube_faces=args.require_all_cube_faces,
        )
        missing_results = target_names - final_result_names
        missing_inputs = target_names - existing_names
        logging.info(
            "Final local coverage: target=%s results=%s nms_inputs=%s missing_results=%s missing_inputs=%s",
            len(target_names),
            len(final_result_names),
            len(existing_names & target_names),
            len(missing_results),
            len(missing_inputs),
        )
        if missing_results or missing_inputs or unreadable:
            raise RuntimeError(
                "Local backfill is incomplete: "
                f"missing_results={sorted(missing_results)}, "
                f"missing_inputs={sorted(missing_inputs)}, "
                f"unreadable_inputs={unreadable}"
            )

        local_nms_run = run_s3_nms(
            s3_client,
            bucket=bucket,
            project_id=project_id,
            use_case_id=use_case_id,
            images_prefix=images_prefix,
            cube_faces=cube_faces,
            remote_config=remote_config,
            run_id=run_id,
            output_root=args.nms_output_root,
            write_s3_report=not args.no_s3_report,
            publish_issues=args.publish_issues,
        )
        log_summary(
            expected_count=len(target_names),
            result_count=len(final_result_names),
            existing_before_count=existing_before_count,
            local_candidate_count=local_candidate_count,
            final_input_count=len(existing_names & target_names),
            tag_count=tag_count,
            failure_count=failure_count,
            force=args.force,
            started_at=started_at,
            finalization=local_nms_run.marker,
        )
    finally:
        write_disabled_remote_config(s3_client, bucket, config_key, disabled_config)
        logging.info("Remote QAQC left disabled; local 3D and local NMS own this workflow.")


if __name__ == "__main__":
    main()
