from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any


FACE_DIRECTIONS = ["up", "front", "right", "back", "left", "down"]
FACE_RE = re.compile(r"^(.+)_(\d+)_(up|front|right|back|left|down)(?:\.[^.]+)?$", re.IGNORECASE)
FACE_BASIS = {
    "up": {
        "normal": {"x": 0, "y": 0, "z": 1},
        "right": {"x": 1, "y": 0, "z": 0},
        "up": {"x": 0, "y": -1, "z": 0},
    },
    "front": {
        "normal": {"x": 0, "y": 1, "z": 0},
        "right": {"x": 1, "y": 0, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
    },
    "right": {
        "normal": {"x": 1, "y": 0, "z": 0},
        "right": {"x": 0, "y": -1, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
    },
    "back": {
        "normal": {"x": 0, "y": -1, "z": 0},
        "right": {"x": -1, "y": 0, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
    },
    "left": {
        "normal": {"x": -1, "y": 0, "z": 0},
        "right": {"x": 0, "y": 1, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
    },
    "down": {
        "normal": {"x": 0, "y": 0, "z": -1},
        "right": {"x": 1, "y": 0, "z": 0},
        "up": {"x": 0, "y": 1, "z": 0},
    },
}


def image_key(image_name: str) -> str:
    return Path(str(image_name)).name.rsplit(".", 1)[0]


def parse_image_name(image_name: str) -> dict:
    key = image_key(image_name)
    match = FACE_RE.match(key)
    if not match:
        raise ValueError(f"Cube image name must look like <pano_id>_<face_index>_<face>.jpg: {image_name}")

    face_index = int(match.group(2))
    face_direction = match.group(3).lower()
    expected = FACE_DIRECTIONS[face_index] if 0 <= face_index < len(FACE_DIRECTIONS) else None
    if expected and expected != face_direction:
        raise ValueError(f"Face index {face_index} should be {expected}, not {face_direction}.")
    return {"imageKey": key, "panoId": match.group(1), "faceIndex": face_index, "faceDirection": face_direction}


def _face_order(cube_faces: dict[str, Any]) -> list[str]:
    cubemap = cube_faces.get("cubemap") or {}
    return cubemap.get("faceOrder") or cubemap.get("face_order") or FACE_DIRECTIONS


def _templated_names(cube_faces: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Names derived from `imageNameTemplate`, for panos that carry no explicit image list."""
    template = cube_faces.get("imageNameTemplate") or cube_faces.get("image_name_template")
    if not isinstance(template, str):
        return {}

    face_order = _face_order(cube_faces)
    derived: dict[str, dict[str, Any]] = {}
    for pano_id in cube_faces.get("panos") or {}:
        for face_index, face_direction in enumerate(face_order):
            try:
                file_name = template.format(
                    pano_id=pano_id,
                    face_index=face_index,
                    face_direction=face_direction,
                )
            except (KeyError, IndexError):
                logging.warning("Ignoring unusable imageNameTemplate %r in cube_faces.json", template)
                return {}
            derived[image_key(file_name)] = {
                "fileName": file_name,
                "faceIndex": face_index,
                "panoId": pano_id,
                "faceDirection": face_direction,
                "templated": True,
            }
    return derived


def image_index(cube_faces: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map image key (file name without extension) to its pano/face metadata.

    Build this once per run and pass it around: it walks every pano on every call.
    """
    index: dict[str, dict[str, Any]] = {}
    face_order = _face_order(cube_faces)

    for pano_id, pano in (cube_faces.get("panos") or {}).items():
        for image in pano.get("images") or []:
            face_index = int(image.get("faceIndex", 0))
            index[image_key(image["fileName"])] = {
                **image,
                "panoId": pano_id,
                "faceDirection": face_order[face_index] if 0 <= face_index < len(face_order) else "",
            }

    for image in cube_faces.get("images") or []:
        info = parse_image_name(image["fileName"])
        index.setdefault(
            info["imageKey"],
            {
                **image,
                "panoId": image.get("panoId") or info["panoId"],
                "faceDirection": image.get("faceDirection") or info["faceDirection"],
            },
        )

    for key, image in _templated_names(cube_faces).items():
        index.setdefault(key, image)
    return index


def expected_image_names(index: dict[str, dict[str, Any]]) -> set[str]:
    """Raw image names QAQC waits for. Explicit entries win; the template is only a fallback."""
    explicit = {str(image["fileName"]) for image in index.values() if not image.get("templated")}
    return explicit or {str(image["fileName"]) for image in index.values()}


def image_meta(cube_faces: dict[str, Any], index: dict[str, dict[str, Any]], image_name: str) -> dict[str, Any]:
    info = parse_image_name(image_name)
    image = index.get(info["imageKey"])
    pano = (cube_faces.get("panos") or {}).get(info["panoId"])
    if not image or not pano:
        raise ValueError(f"{image_name} is not present in cube_faces.json")

    cubemap = cube_faces.get("cubemap") or {}
    overrides = cubemap.get("faceBasis") or cubemap.get("face_basis") or {}
    face_direction = image["faceDirection"]
    return {
        **image,
        "panoId": info["panoId"],
        "pano_position": pano.get("position"),
        "pano_rotation": pano.get("rotation"),
        "face_basis": overrides.get(face_direction) or FACE_BASIS[face_direction],
    }
