from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import unquote

from botocore.exceptions import ClientError


MASTER_USE_CASE_MAP_KEY = "master_use_case_map.json"


@dataclass(frozen=True)
class S3Head:
    content_disposition: str
    last_modified: datetime | None
    version_id: str | None


def read_json(s3_client, bucket: str, key: str) -> dict[str, Any]:
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def put_json(s3_client, bucket: str, key: str, payload: dict[str, Any]):
    return s3_client.put_object(Bucket=bucket, Key=key, Body=(json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"), ContentType="application/json")


def put_json_if_absent(s3_client, bucket: str, key: str, payload: dict[str, Any]) -> bool:
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=(json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"), ContentType="application/json", IfNoneMatch="*")
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code", "")) in {"412", "PreconditionFailed"}:
            return False
        raise
    return True


def head_object(s3_client, bucket: str, key: str) -> S3Head | None:
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code", "")) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    return S3Head(response.get("ContentDisposition", ""), response.get("LastModified"), response.get("VersionId"))


def list_object_keys(s3_client, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        keys.extend(item["Key"] for item in page.get("Contents", []) if item.get("Key"))
    return sorted(keys)


def content_disposition_filename(header: str) -> str | None:
    if not header:
        return None
    candidates: dict[str, str] = {}
    for part in header.split(";"):
        name, _, value = part.strip().partition("=")
        candidates.setdefault(name.lower(), value.strip().strip('"'))
    encoded = candidates.get("filename*")
    if encoded:
        if "''" in encoded:
            encoded = encoded.split("''", 1)[1]
        return unquote(encoded) or None
    return candidates.get("filename") or None


def fetch_master_project_id(s3_client, bucket: str, use_case_id: str) -> str:
    mapping = read_json(s3_client, bucket, MASTER_USE_CASE_MAP_KEY)
    try:
        return str(mapping["master_use_case_map"][use_case_id])
    except KeyError as error:
        raise KeyError(f"use_case_id {use_case_id} is missing from {MASTER_USE_CASE_MAP_KEY}") from error


def master_config_key(master_project_id: str, use_case_id: str) -> str:
    return f"{master_project_id}/use_cases/{use_case_id}/qaqc_config.json"
