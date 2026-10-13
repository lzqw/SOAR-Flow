"""Versioned high-level action semantics and compact replay encodings."""

from __future__ import annotations

import hashlib

import numpy as np


FIXED_SPEED_ACTION_SEMANTICS = "target_point_fixed_speed_v1"
SPEED_COUPLED_ACTION_SEMANTICS = "target_point_speed_coupled_v1"
TARGET_SPEED_LATERAL_ACTION_SEMANTICS = "target_speed_lateral_v1"
SUPPORTED_ACTION_SEMANTICS = frozenset(
    {
        FIXED_SPEED_ACTION_SEMANTICS,
        SPEED_COUPLED_ACTION_SEMANTICS,
        TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
    }
)
ACTION_SEMANTICS_CODE = {
    FIXED_SPEED_ACTION_SEMANTICS: 1,
    SPEED_COUPLED_ACTION_SEMANTICS: 2,
    TARGET_SPEED_LATERAL_ACTION_SEMANTICS: 3,
}
ACTION_SEMANTICS_NAME = {code: name for name, code in ACTION_SEMANTICS_CODE.items()}


def validate_action_semantics_version(value: str) -> str:
    value = str(value)
    if value not in SUPPORTED_ACTION_SEMANTICS:
        raise ValueError(f"unsupported action semantics version: {value}")
    return value


def action_semantics_code(value: str) -> np.int8:
    return np.int8(ACTION_SEMANTICS_CODE[validate_action_semantics_version(value)])


def action_semantics_name(value: int) -> str:
    try:
        return ACTION_SEMANTICS_NAME[int(value)]
    except KeyError as error:
        raise ValueError(f"unsupported action semantics code: {value}") from error


def encode_scene_signature(value: str) -> np.ndarray:
    """Encode a signature as 32 bytes for shape-stable JAX replay."""

    value = str(value).strip().lower()
    try:
        payload = bytes.fromhex(value)
    except ValueError:
        payload = b""
    if len(payload) != 32:
        payload = hashlib.sha256(value.encode("utf-8")).digest()
    return np.frombuffer(payload, dtype=np.uint8).copy()


__all__ = [
    "ACTION_SEMANTICS_CODE",
    "ACTION_SEMANTICS_NAME",
    "FIXED_SPEED_ACTION_SEMANTICS",
    "SPEED_COUPLED_ACTION_SEMANTICS",
    "TARGET_SPEED_LATERAL_ACTION_SEMANTICS",
    "SUPPORTED_ACTION_SEMANTICS",
    "action_semantics_code",
    "action_semantics_name",
    "encode_scene_signature",
    "validate_action_semantics_version",
]
