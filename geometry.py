from __future__ import annotations

import math


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize(vector: dict) -> dict | None:
    length = math.sqrt(sum(float(vector.get(axis, 0.0)) ** 2 for axis in ("x", "y", "z")))
    if not length:
        return None
    return {axis: float(vector.get(axis, 0.0)) / length for axis in ("x", "y", "z")}


def add_scaled(base: dict, delta: dict, scale: float) -> dict:
    return {
        "x": float(base.get("x", 0.0)) + float(delta.get("x", 0.0)) * scale,
        "y": float(base.get("y", 0.0)) + float(delta.get("y", 0.0)) * scale,
        "z": float(base.get("z", 0.0)) + float(delta.get("z", 0.0)) * scale,
    }


def rotate_by_quaternion(vector: dict, quaternion: dict | None) -> dict:
    if not quaternion:
        return vector
    length = math.sqrt(sum(float(quaternion.get(axis, 0.0)) ** 2 for axis in ("x", "y", "z", "w")))
    if not length:
        return vector
    qx = float(quaternion.get("x", 0.0)) / length
    qy = float(quaternion.get("y", 0.0)) / length
    qz = float(quaternion.get("z", 0.0)) / length
    qw = float(quaternion.get("w", 1.0)) / length
    vx = float(vector.get("x", 0.0))
    vy = float(vector.get("y", 0.0))
    vz = float(vector.get("z", 0.0))
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return {
        "x": vx + qw * tx + (qy * tz - qz * ty),
        "y": vy + qw * ty + (qz * tx - qx * tz),
        "z": vz + qw * tz + (qx * ty - qy * tx),
    }


def rotation_for_api_vector(vector: dict) -> dict:
    direction = normalize({"x": vector.get("x", 0.0), "y": vector.get("z", 0.0), "z": -vector.get("y", 0.0)})
    if not direction:
        raise ValueError("Could not build a Matterport viewing direction.")
    return {
        "x": math.degrees(math.asin(clamp(direction["y"], -1.0, 1.0))),
        "y": math.degrees(math.atan2(direction["z"], -direction["x"])),
    }


def rotation_for_bbox_center(coordinates: list[float], image: dict) -> tuple[dict, dict]:
    x, y, width, height = [float(value) for value in coordinates]
    center = {"x": clamp(x + width / 2.0, 0.0, 1.0), "y": clamp(y + height / 2.0, 0.0, 1.0)}
    basis = image.get("face_basis")
    if not basis:
        raise ValueError("Image metadata is missing face_basis.")
    u = 2.0 * center["x"] - 1.0
    v = 1.0 - 2.0 * center["y"]
    local = normalize(add_scaled(add_scaled(basis["normal"], basis["right"], u), basis["up"], v))
    if not local:
        raise ValueError("Could not build a local cube-face vector.")
    return rotation_for_api_vector(rotate_by_quaternion(local, image.get("pano_rotation"))), center


def rotation_for_face_center(image: dict) -> dict:
    basis = image.get("face_basis")
    if not basis:
        raise ValueError("Image metadata is missing face_basis.")
    return rotation_for_api_vector(rotate_by_quaternion(basis["normal"], image.get("pano_rotation")))
