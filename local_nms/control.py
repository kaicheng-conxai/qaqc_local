from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from s3_utils import put_json


def disabled_remote_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a config that disables both remote 3D processing and remote NMS."""
    updated = deepcopy(config)
    updated["enabled"] = False
    updated["nms"] = {**(updated.get("nms") or {}), "enabled": False}
    return updated


def write_disabled_remote_config(
    s3_client,
    bucket: str,
    config_key: str,
    config: dict[str, Any],
) -> None:
    put_json(s3_client, bucket, config_key, config)
    logging.info(
        "Remote QAQC disabled at s3://%s/%s: enabled=%s nms.enabled=%s",
        bucket,
        config_key,
        config.get("enabled"),
        (config.get("nms") or {}).get("enabled"),
    )
