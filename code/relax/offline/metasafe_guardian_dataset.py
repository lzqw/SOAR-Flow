"""Fixed-seed MetaSafe dataset adapter for Guardian-CalQL training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from relax.utils.experience import (
    INTERVENTION_DIAGNOSTIC,
    INTERVENTION_HARD,
    INTERVENTION_NONE,
    INTERVENTION_SOFT,
    GuardianCalQLExperience,
    GuardianDSACExperience,
)
from relax.utils.action_semantics import (
    FIXED_SPEED_ACTION_SEMANTICS,
    SPEED_COUPLED_ACTION_SEMANTICS,
    TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
    action_semantics_code,
    encode_scene_signature,
    validate_action_semantics_version,
)


REQUIRED_KEYS = {
    "obs",
    "next_obs",
    "raw_action",
    "action",
    "teacher_action",
    "reward",
    "done",
    "episode_id",
    "step_index",
    "projection_used",
    "projection_norm",
    "projection_failed",
    "solver_finite",
    "hard_cbf_margin",
    "obstacle_clearance",
    "road_clearance",
    "kinematic_slack",
    "critic_action_compatible",
    "bc_valid",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discounted_return_to_go(
    reward: np.ndarray,
    done: np.ndarray,
    episode_id: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    """Compute behavior-policy RTG without leaking across episodes."""

    reward = np.asarray(reward, dtype=np.float32).reshape(-1)
    done = np.asarray(done, dtype=bool).reshape(-1)
    episode_id = np.asarray(episode_id).reshape(-1)
    if not (reward.shape == done.shape == episode_id.shape):
        raise ValueError("reward/done/episode_id must have the same shape")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    result = np.zeros_like(reward, dtype=np.float32)
    running = 0.0
    next_episode = None
    for index in range(reward.size - 1, -1, -1):
        current_episode = int(episode_id[index])
        if next_episode is None or current_episode != next_episode or done[index]:
            running = 0.0
        running = float(reward[index]) + gamma * running
        result[index] = np.float32(running)
        next_episode = current_episode
    return result


def discounted_guardian_dependence_targets(
    intervention_hard: np.ndarray,
    intervention_soft: np.ndarray,
    intervention_numerical: np.ndarray,
    corrected_valid: np.ndarray,
    done: np.ndarray,
    episode_id: np.ndarray,
    *,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute bounded V2.2 raw/corrected Guardian-dependence returns."""

    hard = np.asarray(intervention_hard, dtype=bool).reshape(-1)
    soft = np.asarray(intervention_soft, dtype=bool).reshape(-1)
    numerical = np.asarray(intervention_numerical, dtype=bool).reshape(-1)
    corrected_valid = np.asarray(corrected_valid, dtype=bool).reshape(-1)
    done = np.asarray(done, dtype=bool).reshape(-1)
    episode_id = np.asarray(episode_id).reshape(-1)
    shapes = {
        value.shape
        for value in (hard, soft, numerical, corrected_valid, done, episode_id)
    }
    if len(shapes) != 1:
        raise ValueError("Guardian-dependence target fields must align")
    if not 0.0 < gamma < 1.0:
        raise ValueError("Guardian-dependence discount must be in (0, 1)")
    if np.any(hard & soft):
        raise ValueError("hard and soft intervention labels overlap")

    raw = np.zeros(hard.shape, dtype=np.float32)
    corrected = np.zeros(hard.shape, dtype=np.float32)
    valid = ~numerical
    corrected_mask = corrected_valid & valid & (hard | soft)
    continuation = 0.0
    next_episode = None
    for index in range(hard.size - 1, -1, -1):
        current_episode = int(episode_id[index])
        if next_episode is None or current_episode != next_episode or done[index]:
            continuation = 0.0
        if numerical[index]:
            continuation = 0.0
            next_episode = current_episode
            continue
        cost = 1.0 if hard[index] else 0.5 if soft[index] else 0.0
        future = float(gamma) * continuation
        raw_target = cost + (1.0 - cost) * future
        corrected_target = future
        raw[index] = np.float32(np.clip(raw_target, 0.0, 1.0))
        if corrected_mask[index]:
            corrected[index] = np.float32(
                np.clip(corrected_target, 0.0, 1.0)
            )
        continuation = float(raw[index])
        next_episode = current_episode
    return raw, corrected, valid, corrected_mask


def split_complete_episodes(
    episode_id: np.ndarray,
    *,
    seed: int = 0,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, np.ndarray]:
    """Return disjoint transition indices split only at episode boundaries."""

    episode_id = np.asarray(episode_id).reshape(-1)
    if train_fraction <= 0.0 or validation_fraction <= 0.0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train + validation fractions must be below one")
    episodes = np.unique(episode_id)
    if episodes.size < 3:
        raise ValueError("at least three episodes are required")
    rng = np.random.default_rng(seed)
    shuffled = episodes.copy()
    rng.shuffle(shuffled)
    train_count = max(int(np.floor(episodes.size * train_fraction)), 1)
    validation_count = max(int(np.floor(episodes.size * validation_fraction)), 1)
    if train_count + validation_count >= episodes.size:
        validation_count = max(episodes.size - train_count - 1, 1)
    train_episodes = shuffled[:train_count]
    validation_episodes = shuffled[train_count : train_count + validation_count]
    test_episodes = shuffled[train_count + validation_count :]
    split_episodes = {
        "train": np.sort(train_episodes),
        "validation": np.sort(validation_episodes),
        "test": np.sort(test_episodes),
    }
    indices = {
        name: np.flatnonzero(np.isin(episode_id, selected))
        for name, selected in split_episodes.items()
    }
    assigned = np.concatenate(list(indices.values()))
    if assigned.size != episode_id.size or np.unique(assigned).size != assigned.size:
        raise AssertionError("episode split lost or duplicated transitions")
    return {**indices, "episodes": split_episodes}


def future_intervention_label(
    intervention: np.ndarray,
    episode_id: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    """Whether an intervention occurs in the current/following H steps."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    intervention = np.asarray(intervention, dtype=bool).reshape(-1)
    episode_id = np.asarray(episode_id).reshape(-1)
    if intervention.shape != episode_id.shape:
        raise ValueError("intervention and episode_id shape mismatch")
    result = np.zeros_like(intervention)
    for episode in np.unique(episode_id):
        indices = np.flatnonzero(episode_id == episode)
        values = intervention[indices]
        # A reverse rolling sum is linear in trajectory length.
        prefix = np.concatenate([[0], np.cumsum(values.astype(np.int32))])
        for local_index in range(indices.size):
            end = min(local_index + horizon, indices.size)
            result[indices[local_index]] = prefix[end] - prefix[local_index] > 0
    return result


@dataclass(frozen=True)
class MetaSafeDatasetSummary:
    path: str
    sha256: str
    transitions: int
    decision_transitions: int
    excluded_reset_transitions: int
    episodes: int
    obs_dim: int
    action_dim: int
    train_episodes: int
    validation_episodes: int
    test_episodes: int
    intervention_rate: float
    hard_intervention_rate: float
    soft_intervention_rate: float
    numerical_intervention_rate: float
    diagnostic_authority_rate: float
    actual_low_level_modification_rate: float
    proxy_pairs: int
    teacher_focus_rows: int
    historical_raw_certificate_available: bool
    canonical_maneuver_side: int
    canonical_train_episodes: int
    canonical_validation_episodes: int
    canonical_test_episodes: int
    episode_step_limit: int | None
    speed_sampling_profile: str | None


class GuardianMetaSafeDataset:
    """Validated in-memory adapter around the 300-episode NPZ dataset."""

    def __init__(
        self,
        path: str | Path,
        *,
        gamma: float = 0.99,
        proxy_discount: float = 0.97,
        split_seed: int = 0,
        soft_pair_weight: float = 0.25,
        soft_teacher_weight: float = 1.0,
        exclude_legacy_reset_step: bool = True,
        expected_sha256: str | None = None,
        expected_action_semantics_version: str | None = None,
        expected_scene_seed: int | None = None,
        expected_scene_signature: str | None = None,
        expected_scene_selection_manifest_hash: str | None = None,
        expected_lookahead_distance: float | None = None,
        expected_max_command_speed: float | None = None,
        expected_speed_mapping: str | None = None,
        expected_mpc_horizon: float | None = None,
        expected_episode_step_limit: int | None = None,
        expected_speed_sampling_profile: str | None = None,
    ):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = sha256_file(self.path)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError(
                f"dataset SHA256 mismatch: {self.sha256} != {expected_sha256}"
            )
        with np.load(self.path, allow_pickle=False) as archive:
            missing = sorted(REQUIRED_KEYS - set(archive.files))
            if missing:
                raise ValueError(f"dataset is missing required keys: {missing}")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        transition_count = int(arrays["obs"].shape[0])
        if transition_count < 1:
            raise ValueError("dataset is empty")
        for name, value in arrays.items():
            if value.shape[0] != transition_count:
                raise ValueError(
                    f"field {name} has {value.shape[0]} rows, expected {transition_count}"
                )

        def uniform_string(name: str, fallback: str | None = None) -> str | None:
            if name not in arrays:
                return fallback
            values = np.unique(arrays[name].astype(str).reshape(-1))
            if values.size != 1:
                raise ValueError(f"dataset contains multiple {name} values")
            return str(values[0])

        def uniform_float(name: str) -> float | None:
            if name not in arrays:
                return None
            values = arrays[name].astype(np.float64).reshape(-1)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"dataset {name} contains non-finite values")
            if not np.allclose(values, values[0], atol=1e-8, rtol=0.0):
                raise ValueError(f"dataset contains multiple {name} values")
            return float(values[0])

        recorded_semantics = uniform_string(
            "action_semantics_version",
            FIXED_SPEED_ACTION_SEMANTICS,
        )
        recorded_semantics = validate_action_semantics_version(
            str(recorded_semantics)
        )
        if expected_action_semantics_version is not None:
            expected_action_semantics_version = (
                validate_action_semantics_version(
                    expected_action_semantics_version
                )
            )
            if "action_semantics_version" not in arrays:
                raise ValueError(
                    "dataset has no action_semantics_version metadata"
                )
            if recorded_semantics != expected_action_semantics_version:
                raise ValueError(
                    "dataset action semantics mismatch: "
                    f"{recorded_semantics} != "
                    f"{expected_action_semantics_version}"
                )
        self.action_semantics_version = recorded_semantics

        v2_2_required = {
            "action_semantics_version",
            "lookahead_distance",
            "max_command_speed",
            "speed_mapping",
            "mpc_horizon",
            "scene_seed",
            "scene_signature",
            "raw_target_speed",
            "executed_target_speed",
            "speed_command_delta",
            "speed_only_intervention",
        }
        if recorded_semantics == SPEED_COUPLED_ACTION_SEMANTICS:
            missing_v2_2 = sorted(v2_2_required - set(arrays))
            if missing_v2_2:
                raise ValueError(
                    "speed-coupled dataset is missing metadata/fields: "
                    f"{missing_v2_2}"
                )
        v2_3_required = {
            "action_semantics_version",
            "min_command_speed",
            "max_command_speed",
            "speed_mapping",
            "mpc_horizon",
            "scene_seed",
            "scene_signature",
            "reachable_speed_lower",
            "reachable_speed_upper",
            "raw_target_speed",
            "executed_target_speed",
            "actual_vehicle_speed",
            "speed_command_delta",
            "emergency_stop",
        }
        if recorded_semantics == TARGET_SPEED_LATERAL_ACTION_SEMANTICS:
            missing_v2_3 = sorted(v2_3_required - set(arrays))
            if missing_v2_3:
                raise ValueError(
                    "target-speed-lateral dataset is missing metadata/fields: "
                    f"{missing_v2_3}"
                )

        scene_seeds = np.unique(
            arrays.get(
                "scene_seed",
                np.zeros((transition_count,), dtype=np.int32),
            ).astype(np.int32)
        )
        if scene_seeds.size != 1:
            raise ValueError("offline dataset must contain exactly one scene seed")
        self.scene_seed = int(scene_seeds[0])
        if (
            expected_scene_seed is not None
            and self.scene_seed != int(expected_scene_seed)
        ):
            raise ValueError(
                f"dataset scene seed mismatch: {self.scene_seed} != "
                f"{int(expected_scene_seed)}"
            )
        self.scene_signature = str(
            uniform_string("scene_signature", "") or ""
        )
        if expected_scene_signature is not None:
            if "scene_signature" not in arrays:
                raise ValueError("dataset has no scene_signature metadata")
            if self.scene_signature != str(expected_scene_signature):
                raise ValueError("dataset scene signature mismatch")
        self.scene_selection_manifest_hash = str(
            uniform_string("scene_selection_manifest_hash", "") or ""
        )
        if expected_scene_selection_manifest_hash is not None:
            if "scene_selection_manifest_hash" not in arrays:
                raise ValueError(
                    "dataset has no scene_selection_manifest_hash metadata"
                )
            if self.scene_selection_manifest_hash != str(
                expected_scene_selection_manifest_hash
            ):
                raise ValueError("dataset scene selection manifest hash mismatch")

        metadata_expectations = {
            "lookahead_distance": expected_lookahead_distance,
            "max_command_speed": expected_max_command_speed,
            "mpc_horizon": expected_mpc_horizon,
        }
        self.action_metadata: dict[str, object] = {
            "action_semantics_version": self.action_semantics_version,
            "scene_seed": self.scene_seed,
            "scene_signature": self.scene_signature,
            "scene_selection_manifest_hash": (
                self.scene_selection_manifest_hash
            ),
        }
        for name, expected in metadata_expectations.items():
            recorded = uniform_float(name)
            self.action_metadata[name] = recorded
            if expected is not None:
                if recorded is None:
                    raise ValueError(f"dataset has no {name} metadata")
                if not np.isclose(recorded, float(expected), atol=1e-8):
                    raise ValueError(
                        f"dataset {name} mismatch: {recorded} != {expected}"
                    )
        recorded_speed_mapping = uniform_string("speed_mapping", None)
        self.action_metadata["speed_mapping"] = recorded_speed_mapping
        if expected_speed_mapping is not None:
            if recorded_speed_mapping is None:
                raise ValueError("dataset has no speed_mapping metadata")
            if recorded_speed_mapping != str(expected_speed_mapping):
                raise ValueError("dataset speed_mapping mismatch")
        recorded_step_limit = uniform_float("episode_step_limit")
        if recorded_step_limit is not None and (
            recorded_step_limit < 1.0
            or not float(recorded_step_limit).is_integer()
        ):
            raise ValueError("dataset episode_step_limit is not a positive integer")
        self.episode_step_limit = (
            None if recorded_step_limit is None else int(recorded_step_limit)
        )
        self.action_metadata["episode_step_limit"] = self.episode_step_limit
        if expected_episode_step_limit is not None:
            if self.episode_step_limit is None:
                raise ValueError("dataset has no episode_step_limit metadata")
            if self.episode_step_limit != int(expected_episode_step_limit):
                raise ValueError(
                    "dataset episode_step_limit mismatch: "
                    f"{self.episode_step_limit} != {int(expected_episode_step_limit)}"
                )
        self.speed_sampling_profile = uniform_string(
            "speed_sampling_profile", None
        )
        self.action_metadata["speed_sampling_profile"] = (
            self.speed_sampling_profile
        )
        if expected_speed_sampling_profile is not None:
            if self.speed_sampling_profile is None:
                raise ValueError("dataset has no speed_sampling_profile metadata")
            if self.speed_sampling_profile != str(expected_speed_sampling_profile):
                raise ValueError(
                    "dataset speed_sampling_profile mismatch: "
                    f"{self.speed_sampling_profile} != "
                    f"{expected_speed_sampling_profile}"
                )
        if not np.all(arrays["critic_action_compatible"]):
            raise ValueError("dataset contains action-incompatible critic rows")
        if not np.all(np.isfinite(arrays["obs"])):
            raise ValueError("dataset observations contain non-finite values")
        if not np.all(np.isfinite(arrays["action"])):
            raise ValueError("dataset executed actions contain non-finite values")
        if not np.all(np.isfinite(arrays["reward"])):
            raise ValueError("dataset rewards contain non-finite values")

        self._source = arrays
        self.episode_id = arrays["episode_id"].astype(np.int32, copy=False)
        self.splits = split_complete_episodes(self.episode_id, seed=split_seed)
        # The historical v1 archive starts policy decisions directly from the
        # observation returned by reset().  When one fixed post-reset frame is
        # used at deployment, row zero of every old episode must not train or
        # validate the Actor/critics.  New collectors mark rows as already
        # post-warmup and therefore retain their recorded step zero.
        if (
            "post_reset_warmup_steps" in arrays
            and np.any(arrays["post_reset_warmup_steps"].reshape(-1) > 0)
        ):
            decision_mask = arrays["post_reset_warmup_steps"].reshape(-1) > 0
        elif exclude_legacy_reset_step:
            decision_mask = arrays["step_index"].reshape(-1) > 0
        else:
            decision_mask = np.ones((transition_count,), dtype=bool)
        if not np.any(decision_mask):
            raise ValueError("dataset has no post-reset policy-decision rows")
        self.decision_mask = decision_mask
        self._decision_indices = {
            name: indices[decision_mask[indices]]
            for name, indices in self.splits.items()
            if name != "episodes"
        }
        projection_used = arrays["projection_used"].astype(bool, copy=False)
        recorded_intervention = arrays.get("intervention", projection_used).astype(
            bool, copy=False
        )
        # New archives explicitly separate actual closed-loop modification from
        # shadow/audit authority.  Historical v1 archives recorded only genuine
        # high-level projection, so their old intervention bit is a valid
        # backward-compatible default.
        actual_control_modified = arrays.get(
            "actual_control_modified", recorded_intervention
        ).astype(bool, copy=False)
        intervention = actual_control_modified
        actual_low_level_modified = arrays.get(
            "actual_low_level_modified", np.zeros_like(intervention)
        ).astype(bool, copy=False)
        audit_authority = arrays.get(
            "audit_authority",
            arrays.get("low_level_cbf_authority", intervention),
        ).astype(bool, copy=False)
        audit_authority |= intervention
        numerical_default = np.logical_or(
            arrays["projection_failed"].astype(bool, copy=False),
            ~arrays["solver_finite"].astype(bool, copy=False),
        )
        numerical = arrays.get(
            "intervention_numerical", numerical_default
        ).astype(bool, copy=False)
        # The v1 collector stored only the executed-action certificate.  It is
        # therefore not valid to infer that a projected raw action crossed a
        # hard boundary.  Preserve those historical pairs as soft evidence.
        raw_certificate_available = all(
            name in arrays
            for name in (
                "raw_hard_cbf_margin",
                "raw_obstacle_clearance",
                "raw_road_clearance",
                "raw_kinematic_slack",
            )
        )
        recorded_kind = arrays.get("intervention_kind")
        if recorded_kind is not None:
            recorded_kind = recorded_kind.astype(np.int8, copy=False)
        if raw_certificate_available:
            # Recompute Hard semantics from physical feasibility instead of
            # trusting archives produced by the former rule that promoted a
            # negative CBF-rate margin alone to Hard.  The latter is only an
            # audit signal when body/road/kinematic margins remain feasible.
            physical_violation = (
                (np.isfinite(arrays["raw_obstacle_clearance"]))
                & (arrays["raw_obstacle_clearance"] < 0.0)
            ) | (
                (np.isfinite(arrays["raw_road_clearance"]))
                & (arrays["raw_road_clearance"] < 0.0)
            ) | (
                (np.isfinite(arrays["raw_kinematic_slack"]))
                & (arrays["raw_kinematic_slack"] < 0.0)
            )
            backup_failure = np.zeros_like(intervention)
            if "backup_feasible" in arrays:
                backup_failure |= ~arrays["backup_feasible"].astype(
                    bool, copy=False
                )
            if "recovery_success" in arrays:
                backup_failure |= ~arrays["recovery_success"].astype(
                    bool, copy=False
                )
            hard = intervention & ~numerical & (
                physical_violation | backup_failure
            )
        elif recorded_kind is not None:
            hard = (recorded_kind == INTERVENTION_HARD) & intervention
        elif "intervention_hard" in arrays:
            hard = arrays["intervention_hard"].astype(bool, copy=False)
        else:
            hard = np.zeros_like(intervention)
        hard &= intervention & ~numerical
        if raw_certificate_available:
            soft = intervention & ~hard & ~numerical
        elif recorded_kind is not None:
            soft = (recorded_kind == INTERVENTION_SOFT) & intervention
        else:
            soft = arrays.get(
                "intervention_soft", intervention & ~hard & ~numerical
            ).astype(bool, copy=False)
        soft &= intervention & ~hard & ~numerical
        diagnostic = np.zeros_like(intervention)
        if recorded_kind is not None:
            diagnostic = recorded_kind == INTERVENTION_DIAGNOSTIC
        else:
            diagnostic = audit_authority & ~intervention
        diagnostic = np.asarray(diagnostic, dtype=bool) | numerical
        intervention_kind = np.full(
            (transition_count,), INTERVENTION_NONE, dtype=np.int8
        )
        intervention_kind[diagnostic] = INTERVENTION_DIAGNOSTIC
        intervention_kind[hard] = INTERVENTION_HARD
        intervention_kind[soft] = INTERVENTION_SOFT
        pair_valid_default = (
            intervention & ~numerical & (arrays["projection_norm"] > 1e-6)
        )
        pair_valid = arrays.get("proxy_pair_valid", pair_valid_default).astype(
            bool, copy=False
        )
        pair_valid &= intervention & ~numerical
        pair_weight_default = np.where(
            hard,
            1.0,
            np.where(soft, max(float(soft_pair_weight), 0.0), 0.0),
        ).astype(np.float32)
        if raw_certificate_available:
            # Do not retain a stale full-weight pair after a CBF-only
            # rejection has been reclassified as Soft.
            pair_weight = pair_weight_default
        else:
            pair_weight = np.maximum(
                arrays.get("proxy_pair_weight", pair_weight_default).astype(
                    np.float32, copy=False
                ),
                0.0,
            )
        actor_controlled = arrays.get(
            "actor_controlled", ~intervention & ~numerical
        ).astype(bool, copy=False)
        actor_controlled &= ~intervention & ~numerical
        teacher_valid = arrays.get("teacher_valid", arrays["bc_valid"]).astype(
            bool, copy=False
        )
        teacher_weight_default = np.where(
            actor_controlled, 1.0, max(float(soft_teacher_weight), 0.0)
        ).astype(np.float32)
        teacher_weight = np.maximum(
            arrays.get("teacher_weight", teacher_weight_default).astype(
                np.float32, copy=False
            ),
            0.0,
        )
        if raw_certificate_available:
            # An intervention label must obey the same Hard/Soft confidence as
            # its Proxy pair.  Non-intervention proactive/ordinary teachers
            # retain their independently recorded weights.
            teacher_cap = np.where(
                hard,
                1.0,
                np.where(soft, max(float(soft_teacher_weight), 0.0), 0.0),
            ).astype(np.float32)
            teacher_weight = np.where(
                intervention,
                np.minimum(teacher_weight, teacher_cap),
                teacher_weight,
            ).astype(np.float32, copy=False)
        teacher_focus = (
            teacher_valid
            & (teacher_weight > 0.0)
            & (
                intervention
                | (
                    np.linalg.norm(
                        arrays["teacher_action"].astype(np.float32, copy=False)
                        - arrays["action"].astype(np.float32, copy=False),
                        axis=-1,
                    )
                    > 1e-6
                )
            )
        )
        # A coherent ordinary-recovery trajectory executes its teacher action
        # directly, so action difference is intentionally zero.  Its explicit
        # source marker keeps those safety-critical rows available for focused
        # BC sampling without corrupting raw/corrected proxy-pair semantics.
        if "teacher_source" in arrays:
            teacher_focus |= (
                arrays["teacher_source"].astype(np.int8, copy=False) == 4
            ) & teacher_valid & (teacher_weight > 0.0)
        self.teacher_focus_mask = teacher_focus
        rtg = discounted_return_to_go(
            arrays["reward"],
            arrays["done"],
            self.episode_id,
            gamma=gamma,
        )
        corrected_actions = arrays.get(
            "proxy_corrected_action",
            arrays["action"],
        ).astype(np.float32, copy=False)
        corrected_target_valid = (
            intervention
            & ~numerical
            & (
                np.linalg.norm(
                    arrays["raw_action"].astype(np.float32, copy=False)
                    - corrected_actions,
                    axis=-1,
                )
                > 1e-6
            )
        )
        (
            guardian_dependence_target,
            guardian_dependence_corrected_target,
            guardian_dependence_target_valid,
            guardian_dependence_corrected_valid,
        ) = discounted_guardian_dependence_targets(
            hard,
            soft,
            numerical,
            corrected_target_valid,
            arrays["done"],
            self.episode_id,
            gamma=proxy_discount,
        )
        scene_signature_values = arrays.get(
            "scene_signature",
            np.full((transition_count,), self.scene_signature),
        ).astype(str)
        scene_signature_encoded = np.stack(
            [encode_scene_signature(value) for value in scene_signature_values],
            axis=0,
        )
        semantics_codes = np.full(
            (transition_count,),
            action_semantics_code(self.action_semantics_version),
            dtype=np.int8,
        )
        hard_margin = arrays["hard_cbf_margin"].astype(np.float32, copy=False)
        finite_margin = np.abs(hard_margin[np.isfinite(hard_margin)])
        margin_scale = (
            float(np.quantile(finite_margin, 0.90)) if finite_margin.size else 1.0
        )
        margin_scale = max(margin_scale, 1e-6)

        def finite_float(name: str, fallback: float = 0.0) -> np.ndarray:
            return np.nan_to_num(
                arrays[name].astype(np.float32, copy=False),
                nan=fallback,
                posinf=fallback,
                neginf=fallback,
            )

        self.fields: dict[str, np.ndarray] = {
            "obs": arrays["obs"].astype(np.float32, copy=False),
            "action": arrays["action"].astype(np.float32, copy=False),
            "reward": arrays["reward"].astype(np.float32, copy=False),
            "done": arrays["done"].astype(bool, copy=False),
            "next_obs": arrays["next_obs"].astype(np.float32, copy=False),
            "raw_action": arrays["raw_action"].astype(np.float32, copy=False),
            "proxy_corrected_action": arrays.get(
                "proxy_corrected_action", arrays["action"]
            ).astype(np.float32, copy=False),
            "teacher_action": arrays["teacher_action"].astype(np.float32, copy=False),
            "teacher_valid": teacher_valid,
            "teacher_weight": teacher_weight,
            "actor_controlled": actor_controlled,
            "intervention": intervention,
            "intervention_hard": hard,
            "intervention_soft": soft,
            "intervention_numerical": numerical,
            "intervention_kind": intervention_kind,
            "actual_control_modified": actual_control_modified,
            "actual_low_level_modified": actual_low_level_modified,
            "audit_authority": audit_authority,
            "proposal_evaluated": arrays.get(
                "proposal_evaluated", arrays["solver_finite"]
            ).astype(bool, copy=False),
            "proxy_pair_valid": pair_valid,
            "proxy_pair_weight": pair_weight,
            # Corrected actions do not inherit the rejected raw proposal's
            # immediate intervention label.  Both targets are reconstructed
            # backward from the complete episode; V2.2 has no P-TD objective.
            "proxy_cost": np.zeros((transition_count,), dtype=np.float32),
            "guardian_dependence_target": guardian_dependence_target,
            "guardian_dependence_corrected_target": (
                guardian_dependence_corrected_target
            ),
            "guardian_dependence_target_valid": (
                guardian_dependence_target_valid
            ),
            "guardian_dependence_corrected_valid": (
                guardian_dependence_corrected_valid
            ),
            "behavior_return_to_go": rtg,
            # This adapter reconstructs RTG from complete offline episodes, so
            # a numerically zero return remains an explicitly valid target.
            "behavior_return_valid": np.ones((transition_count,), dtype=bool),
            # The fixed offline archive is the trusted support for the frozen
            # Cal-QL reference used by online Q anchoring.
            "q_anchor_valid": np.ones((transition_count,), dtype=bool),
            "raw_target_speed": arrays.get(
                "raw_target_speed",
                np.zeros((transition_count,), dtype=np.float32),
            ).astype(np.float32, copy=False),
            "executed_target_speed": arrays.get(
                "executed_target_speed",
                np.zeros((transition_count,), dtype=np.float32),
            ).astype(np.float32, copy=False),
            "speed_command_delta": arrays.get(
                "speed_command_delta",
                np.zeros((transition_count,), dtype=np.float32),
            ).astype(np.float32, copy=False),
            "speed_only_intervention": arrays.get(
                "speed_only_intervention",
                np.zeros((transition_count,), dtype=bool),
            ).astype(bool, copy=False),
            "action_semantics_version": semantics_codes,
            "scene_seed": arrays.get(
                "scene_seed",
                np.zeros((transition_count,), dtype=np.int32),
            ).astype(np.int32, copy=False),
            "scene_signature": scene_signature_encoded,
            "filter_version": arrays.get(
                "filter_version", np.zeros((transition_count,), dtype=np.int32)
            ).astype(np.int32, copy=False),
            "spatial_reserve": np.nan_to_num(
                arrays.get(
                    "spatial_reserve",
                    np.full((transition_count,), np.nan, dtype=np.float32),
                ).astype(np.float32, copy=False),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            "kinematic_reserve": np.nan_to_num(
                arrays.get(
                    "kinematic_reserve",
                    np.full((transition_count,), np.nan, dtype=np.float32),
                ).astype(np.float32, copy=False),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            "execution_layer": arrays.get(
                "execution_layer", np.where(intervention, 1, 0)
            ).astype(np.int8, copy=False),
            "projection_norm": finite_float("projection_norm"),
            "hard_cbf_margin": finite_float("hard_cbf_margin"),
            "cbf_margin_normalized": np.tanh(
                finite_float("hard_cbf_margin") / margin_scale
            ).astype(np.float32),
            "obstacle_clearance": finite_float("obstacle_clearance"),
            "road_clearance": finite_float("road_clearance"),
            "kinematic_slack": finite_float("kinematic_slack"),
            "solver_finite": arrays["solver_finite"].astype(bool, copy=False),
            "backup_feasible": arrays.get(
                "backup_feasible",
                arrays["solver_finite"].astype(bool, copy=False)
                & ~arrays["projection_failed"].astype(bool, copy=False),
            ).astype(bool, copy=False),
            "recovery_success": arrays.get(
                "recovery_success",
                arrays["solver_finite"].astype(bool, copy=False)
                & ~arrays["projection_failed"].astype(bool, copy=False),
            ).astype(bool, copy=False),
        }
        self.rng = np.random.default_rng(split_seed + 10_007)
        self._pair_indices = {
            name: indices[self.fields["proxy_pair_valid"][indices]]
            for name, indices in self._decision_indices.items()
        }
        self._teacher_indices = {
            name: indices[teacher_focus[indices]]
            for name, indices in self._decision_indices.items()
        }
        episodes = self.splits["episodes"]
        maneuver_side: dict[int, int] = {}
        for episode in np.unique(self.episode_id):
            episode_mask = self.episode_id == episode
            projected = episode_mask & pair_valid
            lateral = arrays["action"][projected, 1]
            mean_lateral = float(np.mean(lateral)) if lateral.size else 0.0
            maneuver_side[int(episode)] = int(
                1 if mean_lateral > 1e-3 else -1 if mean_lateral < -1e-3 else 0
            )
        train_sides = np.asarray(
            [maneuver_side[int(episode)] for episode in episodes["train"]],
            dtype=np.int8,
        )
        positive = int(np.count_nonzero(train_sides > 0))
        negative = int(np.count_nonzero(train_sides < 0))
        canonical_side = (
            1 if positive >= negative and positive else -1 if negative else 0
        )
        episode_side_array = np.asarray(
            [maneuver_side[int(episode)] for episode in self.episode_id],
            dtype=np.int8,
        )
        self.canonical_maneuver_side = canonical_side
        self.canonical_transition_mask = (
            (episode_side_array == canonical_side) & decision_mask
        )
        self._canonical_indices = {
            name: indices[self.canonical_transition_mask[indices]]
            for name, indices in self._decision_indices.items()
        }
        self._canonical_pair_indices = {
            name: indices[self.fields["proxy_pair_valid"][indices]]
            for name, indices in self._canonical_indices.items()
        }
        self._canonical_teacher_indices = {
            name: indices[teacher_focus[indices]]
            for name, indices in self._canonical_indices.items()
        }
        canonical_episode_counts = {
            name: int(
                np.count_nonzero(
                    [
                        maneuver_side[int(episode)] == canonical_side
                        for episode in selected
                    ]
                )
            )
            for name, selected in episodes.items()
        }
        self.summary = MetaSafeDatasetSummary(
            path=str(self.path),
            sha256=self.sha256,
            transitions=transition_count,
            decision_transitions=int(np.count_nonzero(decision_mask)),
            excluded_reset_transitions=int(np.count_nonzero(~decision_mask)),
            episodes=int(np.unique(self.episode_id).size),
            obs_dim=int(self.fields["obs"].shape[1]),
            action_dim=int(self.fields["action"].shape[1]),
            train_episodes=int(episodes["train"].size),
            validation_episodes=int(episodes["validation"].size),
            test_episodes=int(episodes["test"].size),
            intervention_rate=float(np.mean(intervention[decision_mask])),
            hard_intervention_rate=float(np.mean(hard[decision_mask])),
            soft_intervention_rate=float(np.mean(soft[decision_mask])),
            numerical_intervention_rate=float(np.mean(numerical[decision_mask])),
            diagnostic_authority_rate=float(
                np.mean((diagnostic & ~intervention)[decision_mask])
            ),
            actual_low_level_modification_rate=float(
                np.mean(actual_low_level_modified[decision_mask])
            ),
            proxy_pairs=int(np.count_nonzero(pair_valid & decision_mask)),
            teacher_focus_rows=int(
                np.count_nonzero(teacher_focus & decision_mask)
            ),
            historical_raw_certificate_available=raw_certificate_available,
            canonical_maneuver_side=canonical_side,
            canonical_train_episodes=canonical_episode_counts["train"],
            canonical_validation_episodes=canonical_episode_counts["validation"],
            canonical_test_episodes=canonical_episode_counts["test"],
            episode_step_limit=self.episode_step_limit,
            speed_sampling_profile=self.speed_sampling_profile,
        )

    def __len__(self) -> int:
        return self.summary.decision_transitions

    def experience(self, indices: np.ndarray) -> GuardianCalQLExperience:
        indices = np.asarray(indices, dtype=np.int64)
        return GuardianCalQLExperience(
            **{name: values[indices] for name, values in self.fields.items()}
        )

    def sample(
        self,
        batch_size: int,
        *,
        split: str = "train",
        pair_fraction: float = 0.0,
        teacher_fraction: float = 0.0,
        canonical_only: bool = False,
        to_jax: bool = True,
    ) -> GuardianCalQLExperience:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"unknown split: {split}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= pair_fraction < 1.0:
            raise ValueError("pair_fraction must be in [0, 1)")
        if not 0.0 <= teacher_fraction < 1.0:
            raise ValueError("teacher_fraction must be in [0, 1)")
        if pair_fraction + teacher_fraction >= 1.0:
            raise ValueError("pair_fraction + teacher_fraction must be below 1")
        pair_count = min(int(round(batch_size * pair_fraction)), batch_size)
        teacher_count = min(
            int(round(batch_size * teacher_fraction)),
            batch_size - pair_count,
        )
        ordinary_count = batch_size - pair_count - teacher_count
        ordinary_pool = (
            self._canonical_indices[split]
            if canonical_only
            else self._decision_indices[split]
        )
        if ordinary_pool.size == 0:
            raise ValueError(f"split {split} contains no canonical maneuver rows")
        selected = [
            self.rng.choice(ordinary_pool, size=ordinary_count, replace=True)
        ]
        if pair_count:
            pair_pool = (
                self._canonical_pair_indices[split]
                if canonical_only
                else self._pair_indices[split]
            )
            if pair_pool.size == 0:
                raise ValueError(f"split {split} contains no proxy pairs")
            selected.append(
                self.rng.choice(pair_pool, size=pair_count, replace=True)
            )
        if teacher_count:
            teacher_pool = (
                self._canonical_teacher_indices[split]
                if canonical_only
                else self._teacher_indices[split]
            )
            if teacher_pool.size == 0:
                raise ValueError(f"split {split} contains no focused teacher rows")
            selected.append(
                self.rng.choice(teacher_pool, size=teacher_count, replace=True)
            )
        if len(selected) > 1:
            indices = np.concatenate(selected)
            self.rng.shuffle(indices)
        else:
            indices = selected[0]
        batch = self.experience(indices)
        if to_jax:
            import jax

            batch = jax.device_put(batch)
        return batch

    def iter_split(
        self, split: str, *, batch_size: int, to_jax: bool = True
    ) -> Iterator[GuardianCalQLExperience]:
        indices = self._decision_indices[split]
        for start in range(0, indices.size, batch_size):
            batch = self.experience(indices[start : start + batch_size])
            if to_jax:
                import jax

                batch = jax.device_put(batch)
            yield batch

    def source(self, name: str) -> np.ndarray:
        return self._source[name]

    def split_episode_ids(self) -> Mapping[str, np.ndarray]:
        return self.splits["episodes"]

    def split_transition_indices(
        self, split: str, *, decisions_only: bool = True
    ) -> np.ndarray:
        """Return indices aligned with :meth:`iter_split`.

        Keeping this public avoids accidentally pairing post-reset batches
        with the legacy split indices that still contain one reset row per
        episode.
        """

        if split not in ("train", "validation", "test"):
            raise ValueError(f"unknown split: {split}")
        source = self._decision_indices if decisions_only else self.splits
        return np.asarray(source[split], dtype=np.int64).copy()


class GuardianDSACMetaSafeDataset(GuardianMetaSafeDataset):
    """Fail-closed adapter for the V2.3 explicit-speed replay contract."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_action_semantics_version: str = (
            TARGET_SPEED_LATERAL_ACTION_SEMANTICS
        ),
        expected_min_command_speed: float = 1.0,
        expected_episode_step_limit: int | None = 1200,
        expected_speed_sampling_profile: str | None = "fast_v231",
        **kwargs,
    ):
        if (
            str(expected_action_semantics_version)
            != TARGET_SPEED_LATERAL_ACTION_SEMANTICS
        ):
            raise ValueError(
                "Guardian-DSAC only accepts target_speed_lateral_v1 data"
            )
        super().__init__(
            path,
            expected_action_semantics_version=(
                TARGET_SPEED_LATERAL_ACTION_SEMANTICS
            ),
            expected_episode_step_limit=expected_episode_step_limit,
            expected_speed_sampling_profile=expected_speed_sampling_profile,
            **kwargs,
        )
        arrays = self._source
        count = int(self.summary.transitions)

        def vector(name: str, dtype=np.float32) -> np.ndarray:
            value = np.asarray(arrays[name], dtype=dtype).reshape(count)
            return value

        lower = vector("reachable_speed_lower")
        upper = vector("reachable_speed_upper")
        raw_speed = vector("raw_target_speed")
        executed_speed = vector("executed_target_speed")
        actual_speed = vector("actual_vehicle_speed")
        delta = vector("speed_command_delta")
        emergency = vector("emergency_stop", bool)
        numeric = (lower, upper, raw_speed, executed_speed, actual_speed, delta)
        if not all(np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("V2.3 speed fields contain non-finite values")
        if np.any(upper < lower):
            raise ValueError("reachable speed upper bound is below lower bound")
        if np.any((raw_speed < lower - 2e-5) | (raw_speed > upper + 2e-5)):
            raise ValueError("raw target speed falls outside its reachable interval")
        ordinary = ~emergency
        if np.any(
            ordinary
            & ((executed_speed < lower - 2e-5) | (executed_speed > upper + 2e-5))
        ):
            raise ValueError(
                "non-emergency executed speed falls outside reachable interval"
            )
        if np.any(emergency & (executed_speed > lower + 2e-5)):
            raise ValueError("emergency-stop rows do not reduce the speed command")
        if not np.allclose(
            delta,
            executed_speed - raw_speed,
            atol=2e-5,
            rtol=0.0,
        ):
            raise ValueError("speed_command_delta is inconsistent")
        recorded_min = np.asarray(
            arrays["min_command_speed"], dtype=np.float64
        ).reshape(-1)
        if not np.allclose(
            recorded_min,
            float(expected_min_command_speed),
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError("dataset min_command_speed mismatch")

        self.dsac_fields = {
            **self.fields,
            "reachable_speed_lower": lower,
            "reachable_speed_upper": upper,
            "actual_vehicle_speed": actual_speed,
            "emergency_stop": emergency,
        }

    def experience(self, indices: np.ndarray) -> GuardianDSACExperience:
        indices = np.asarray(indices, dtype=np.int64)
        return GuardianDSACExperience(
            **{name: values[indices] for name, values in self.dsac_fields.items()}
        )

    def speed_coverage(self, *, decisions_only: bool = True) -> dict[str, object]:
        indices = (
            np.flatnonzero(self.decision_mask)
            if decisions_only
            else np.arange(self.summary.transitions)
        )
        lower = self.dsac_fields["reachable_speed_lower"][indices]
        upper = self.dsac_fields["reachable_speed_upper"][indices]
        speed = self.dsac_fields["executed_target_speed"][indices]
        emergency = self.dsac_fields["emergency_stop"][indices]
        fraction = np.clip((speed - lower) / (upper - lower + 1e-6), 0.0, 1.0)
        bins = np.minimum((fraction * 4.0).astype(np.int32), 3)
        counts = np.bincount(bins[~emergency], minlength=4)
        total = int(np.sum(counts))
        return {
            "counts": counts.astype(int).tolist(),
            "fractions": (counts / max(total, 1)).astype(float).tolist(),
            "all_bins_visited": bool(np.all(counts > 0)),
            "highest_bin_fraction": float(counts[3] / max(total, 1)),
            "highest_bin_within_70_percent": bool(
                counts[3] / max(total, 1) <= 0.70
            ),
            "emergency_stop_count": int(np.count_nonzero(emergency)),
        }


__all__ = [
    "GuardianMetaSafeDataset",
    "GuardianDSACMetaSafeDataset",
    "MetaSafeDatasetSummary",
    "discounted_guardian_dependence_targets",
    "discounted_return_to_go",
    "future_intervention_label",
    "sha256_file",
    "split_complete_episodes",
]
