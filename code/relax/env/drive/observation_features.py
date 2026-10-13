"""Pure NumPy observation features for explicit Safe100 obstacles."""

from __future__ import annotations

import numpy as np


EXPLICIT_OBSTACLE_FEATURE_COUNT = 6


def encode_explicit_obstacle_features(
    active_obstacles_ego: np.ndarray,
    *,
    target_lateral_scale: float,
    position_scale: float,
) -> np.ndarray:
    """Encode nearest left/center/right obstacles in six bounded features.

    The first three values are normalized nearest-ahead distances; the last
    three are free-lane flags.  An empty scene is all ones, matching
    MetaDrive's native no-lidar-return convention.
    """

    features = np.ones((EXPLICIT_OBSTACLE_FEATURE_COUNT,), dtype=np.float32)
    obstacles = np.asarray(active_obstacles_ego, dtype=np.float32)
    if obstacles.size == 0:
        return features
    obstacles = obstacles.reshape(-1, obstacles.shape[-1])
    if obstacles.shape[1] < 2:
        return features
    xy = obstacles[:, :2]
    xy = xy[np.all(np.isfinite(xy), axis=1) & (xy[:, 0] >= -2.0)]
    if xy.size == 0:
        return features

    lateral_threshold = 0.5 * max(abs(float(target_lateral_scale)), 1e-3)
    lane_masks = (
        xy[:, 1] > lateral_threshold,
        np.abs(xy[:, 1]) <= lateral_threshold,
        xy[:, 1] < -lateral_threshold,
    )
    distance_scale = max(2.0 * abs(float(position_scale)), 1.0)
    for index, mask in enumerate(lane_masks):
        if np.any(mask):
            nearest = float(np.min(np.maximum(xy[mask, 0], 0.0)))
            features[index] = np.float32(
                np.clip(nearest / distance_scale, 0.0, 1.0)
            )
            features[index + 3] = 0.0
    return features


def world_obstacles_to_ego(
    world_positions: np.ndarray,
    *,
    ego_position: np.ndarray,
    ego_heading: float,
) -> np.ndarray:
    positions = np.asarray(world_positions, dtype=np.float32).reshape(-1, 2)
    if positions.size == 0:
        return positions
    delta = positions - np.asarray(ego_position, dtype=np.float32).reshape(1, 2)
    cosine = float(np.cos(-float(ego_heading)))
    sine = float(np.sin(-float(ego_heading)))
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float32
    )
    return delta @ rotation.T


def inject_explicit_obstacle_features(
    observation: np.ndarray, features: np.ndarray
) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32).reshape(-1).copy()
    features = np.asarray(features, dtype=np.float32).reshape(-1)
    if features.size != EXPLICIT_OBSTACLE_FEATURE_COUNT:
        raise ValueError("unexpected explicit-obstacle feature count")
    if observation.size < features.size:
        raise ValueError("observation is too small for explicit obstacles")
    observation[-features.size :] = features
    return observation
