from __future__ import annotations

import re


def sanitize_issue_id_part(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\.[A-Za-z0-9]+$", "", text)
    text = re.sub(r"[^A-Za-z0-9_]+", "", text)
    return text or "unknown"


def format_issue_id(
    project_name: object,
    image_name: object,
    issue_type: object,
    issue_number: int,
    padding: int = 4,
) -> str:
    return "-".join(
        [
            sanitize_issue_id_part(project_name),
            sanitize_issue_id_part(image_name),
            sanitize_issue_id_part(issue_type),
            str(int(issue_number)).zfill(int(padding)),
        ]
    )
