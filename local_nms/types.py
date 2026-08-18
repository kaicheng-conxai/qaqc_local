from __future__ import annotations

from typing import TypedDict


class Vector2D(TypedDict):
    x: float
    y: float


class Vector3D(TypedDict):
    x: float
    y: float
    z: float


class Location(TypedDict):
    id: int
    panoId: str
    label: str
    rotation: Vector2D
    anchorPosition: Vector3D
    stemVector: Vector3D


class IssueInfo(TypedDict):
    status: str
    created_by: str
    type: str
    trade: str
    severity: str
    reasoning: str
    fe_issue_id: str
    location: Location
    images: list[str]
    task_id: int
    local_instance_ids: list[int]
