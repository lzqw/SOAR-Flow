from typing import NamedTuple, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax


# Stable on-disk codes used by collectors, replay and relabeling tools.
INTERVENTION_NONE = np.int8(0)
INTERVENTION_HARD = np.int8(1)
INTERVENTION_SOFT = np.int8(2)
INTERVENTION_DIAGNOSTIC = np.int8(3)


# ``hard_cbf_margin`` is intentionally absent.  A negative discrete CBF-rate
# certificate can be conservative even while the predicted vehicle body,
# road corridor, and kinematics all remain feasible.  Treating that scalar as
# a physical violation would give conservative CBF-only corrections the same
# learning weight as a collision/road/actuator violation.
PHYSICAL_CERTIFICATE_FIELDS = (
    "obstacle_clearance",
    "road_clearance",
    "kinematic_slack",
)


def certificate_has_physical_violation(summary) -> bool:
    """Return whether a finite geometric/kinematic certificate is violated.

    ``hard_cbf_margin`` remains an audit signal, but is not by itself evidence
    that an action is physically infeasible.  Missing/non-finite fields are
    handled by the separate numerical/solver classification.
    """

    return any(
        np.isfinite(value := float(summary.get(name, np.nan))) and value < 0.0
        for name in PHYSICAL_CERTIFICATE_FIELDS
    )


def probe_batch_size(reward: "jax.Array") -> Optional[int]:
    try:
        if reward.ndim > 0:
            return reward.shape[0]
        else:
            return None
    except AttributeError:
        return None


class Experience(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def __repr__(self):
        return f"Experience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return Experience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool_),
        )

    @staticmethod
    def create(obs, action, reward, terminated, truncated, next_obs, info=None):
        return Experience(
            obs=obs,
            action=action,
            reward=reward,
            done=terminated or truncated,
            next_obs=next_obs,
        )


class JaxSafeSACExperience(NamedTuple):
    """Replay transition used by the JAX hybrid Safe-SAC implementation.

    The critic is trained on the action that was actually executed by the
    certified high-level controller.  ``raw_action`` is kept separately for
    intervention diagnostics.  Teacher fields are explicit so ordinary SAC
    samples and Safe-DAgger samples can coexist in one shape-stable JAX
    batch without Python-side filtering or fake zero context.
    """

    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    raw_action: "jax.Array"
    next_mode: "jax.Array"
    next_recovery_action: "jax.Array"
    actor_controlled: "jax.Array"
    teacher_action: "jax.Array"
    teacher_valid: "jax.Array"
    teacher_weight: "jax.Array"
    intervention: "jax.Array"
    projection_norm: "jax.Array"
    hard_cbf_margin: "jax.Array"

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def __repr__(self):
        return f"JaxSafeSACExperience(size={self.batch_size()})"

    @staticmethod
    def create_example(
        obs_dim: int,
        action_dim: int,
        batch_size: Optional[int] = None,
    ):
        leading_dims = (batch_size,) if batch_size is not None else ()
        scalar = lambda dtype=np.float32: np.zeros(leading_dims, dtype=dtype)
        return JaxSafeSACExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=scalar(),
            done=scalar(np.bool_),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            raw_action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            next_mode=scalar(np.int8),
            next_recovery_action=np.zeros(
                (*leading_dims, action_dim), dtype=np.float32
            ),
            actor_controlled=scalar(np.bool_),
            teacher_action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            teacher_valid=scalar(np.bool_),
            teacher_weight=scalar(),
            intervention=scalar(np.bool_),
            projection_norm=scalar(),
            hard_cbf_margin=scalar(),
        )

    @staticmethod
    def create(
        obs,
        raw_action,
        reward,
        terminated,
        truncated,
        next_obs,
        info: dict,
    ):
        """Build one replay item from ``HybridCBFMPCEnv`` diagnostics."""

        hybrid = info.get("hybrid_safety", {})
        raw_action = np.asarray(
            hybrid.get("raw_action", raw_action), dtype=np.float32
        ).reshape(-1)
        executed_action = np.asarray(
            hybrid.get("executed_action", raw_action), dtype=np.float32
        ).reshape(raw_action.shape)
        next_recovery_action = np.asarray(
            hybrid.get("next_recovery_action", executed_action),
            dtype=np.float32,
        ).reshape(raw_action.shape)
        teacher_action = np.asarray(
            hybrid.get("teacher_action", executed_action), dtype=np.float32
        ).reshape(raw_action.shape)
        teacher_valid = bool(hybrid.get("teacher_valid", False))
        teacher_weight = float(hybrid.get("teacher_weight", float(teacher_valid)))
        intervention = bool(
            hybrid.get("intervention", False)
            or not hybrid.get("actor_controlled", True)
            or float(hybrid.get("projection_norm", 0.0)) > 1e-6
        )
        hard_margin = float(
            hybrid.get(
                "hard_cbf_margin",
                hybrid.get("margin", np.nan),
            )
        )
        return JaxSafeSACExperience(
            obs=np.asarray(obs, dtype=np.float32),
            action=executed_action,
            reward=np.float32(reward),
            done=np.bool_(bool(terminated or truncated)),
            next_obs=np.asarray(next_obs, dtype=np.float32),
            raw_action=raw_action,
            next_mode=np.int8(hybrid.get("next_mode", 0)),
            next_recovery_action=next_recovery_action,
            actor_controlled=np.bool_(hybrid.get("actor_controlled", True)),
            teacher_action=teacher_action,
            teacher_valid=np.bool_(teacher_valid),
            teacher_weight=np.float32(max(teacher_weight, 0.0)),
            intervention=np.bool_(intervention),
            projection_norm=np.float32(hybrid.get("projection_norm", 0.0)),
            hard_cbf_margin=np.float32(hard_margin),
        )


class GuardianCalQLExperience(NamedTuple):
    """Shape-stable transition for guardian-calibrated offline-to-online RL.

    ``action`` is always the action that caused the recorded transition.
    ``raw_action`` is the learner/guardian proposal before filtering and has
    no Bellman successor when it differs from ``action``.
    ``proxy_corrected_action`` is the state-aligned action used only as the
    safer side of the intervention-ranking pair.  It is separate because an
    inverse-MPC teacher can reproduce a protected low-level trajectory without
    itself being the action that caused the recorded transition.  Task TD
    therefore always uses ``action`` and never this counterfactual label.

    Certificate fields deliberately remain separate instead of being summed:
    obstacle distance, road distance and kinematic slack have different units
    and must not be presented as one fictitious physical margin.
    """

    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    raw_action: "jax.Array"
    proxy_corrected_action: "jax.Array"
    teacher_action: "jax.Array"
    teacher_valid: "jax.Array"
    teacher_weight: "jax.Array"
    actor_controlled: "jax.Array"
    # ``intervention`` is deliberately the *actual* control-modification flag.
    # Shadow certificates and CBF-on/off audits that do not modify the executed
    # target/trajectory are represented by ``audit_authority`` and
    # ``intervention_kind == 3`` instead of being promoted to interventions.
    intervention: "jax.Array"
    intervention_hard: "jax.Array"
    intervention_soft: "jax.Array"
    intervention_numerical: "jax.Array"
    intervention_kind: "jax.Array"
    actual_control_modified: "jax.Array"
    actual_low_level_modified: "jax.Array"
    audit_authority: "jax.Array"
    proposal_evaluated: "jax.Array"
    proxy_pair_valid: "jax.Array"
    proxy_pair_weight: "jax.Array"
    proxy_cost: "jax.Array"
    guardian_dependence_target: "jax.Array"
    guardian_dependence_corrected_target: "jax.Array"
    guardian_dependence_target_valid: "jax.Array"
    guardian_dependence_corrected_valid: "jax.Array"
    behavior_return_to_go: "jax.Array"
    # A return of exactly zero is valid.  Never infer validity from the
    # numerical value of ``behavior_return_to_go``.
    behavior_return_valid: "jax.Array"
    # Only trusted offline/reference rows may be pulled toward the frozen
    # offline Q.  Fresh online states deliberately leave this false.
    q_anchor_valid: "jax.Array"
    raw_target_speed: "jax.Array"
    executed_target_speed: "jax.Array"
    speed_command_delta: "jax.Array"
    speed_only_intervention: "jax.Array"
    # Version strings and scene signatures use compact numeric encodings in
    # JAX replay; human-readable values remain in dataset/run metadata.
    action_semantics_version: "jax.Array"
    scene_seed: "jax.Array"
    scene_signature: "jax.Array"
    filter_version: "jax.Array"
    spatial_reserve: "jax.Array"
    kinematic_reserve: "jax.Array"
    execution_layer: "jax.Array"
    projection_norm: "jax.Array"
    hard_cbf_margin: "jax.Array"
    cbf_margin_normalized: "jax.Array"
    obstacle_clearance: "jax.Array"
    road_clearance: "jax.Array"
    kinematic_slack: "jax.Array"
    solver_finite: "jax.Array"
    backup_feasible: "jax.Array"
    recovery_success: "jax.Array"

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def __repr__(self):
        return f"GuardianCalQLExperience(size={self.batch_size()})"

    @staticmethod
    def create_example(
        obs_dim: int,
        action_dim: int,
        batch_size: Optional[int] = None,
    ):
        leading_dims = (batch_size,) if batch_size is not None else ()
        scalar = lambda dtype=np.float32: np.zeros(leading_dims, dtype=dtype)
        return GuardianCalQLExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=scalar(),
            done=scalar(np.bool_),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            raw_action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            proxy_corrected_action=np.zeros(
                (*leading_dims, action_dim), dtype=np.float32
            ),
            teacher_action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            teacher_valid=scalar(np.bool_),
            teacher_weight=scalar(),
            actor_controlled=scalar(np.bool_),
            intervention=scalar(np.bool_),
            intervention_hard=scalar(np.bool_),
            intervention_soft=scalar(np.bool_),
            intervention_numerical=scalar(np.bool_),
            intervention_kind=scalar(np.int8),
            actual_control_modified=scalar(np.bool_),
            actual_low_level_modified=scalar(np.bool_),
            audit_authority=scalar(np.bool_),
            proposal_evaluated=scalar(np.bool_),
            proxy_pair_valid=scalar(np.bool_),
            proxy_pair_weight=scalar(),
            proxy_cost=scalar(),
            guardian_dependence_target=scalar(),
            guardian_dependence_corrected_target=scalar(),
            guardian_dependence_target_valid=scalar(np.bool_),
            guardian_dependence_corrected_valid=scalar(np.bool_),
            behavior_return_to_go=scalar(),
            behavior_return_valid=scalar(np.bool_),
            q_anchor_valid=scalar(np.bool_),
            raw_target_speed=scalar(),
            executed_target_speed=scalar(),
            speed_command_delta=scalar(),
            speed_only_intervention=scalar(np.bool_),
            action_semantics_version=scalar(np.int8),
            scene_seed=scalar(np.int32),
            scene_signature=np.zeros((*leading_dims, 32), dtype=np.uint8),
            filter_version=scalar(np.int32),
            spatial_reserve=scalar(),
            kinematic_reserve=scalar(),
            execution_layer=scalar(np.int8),
            projection_norm=scalar(),
            hard_cbf_margin=scalar(),
            cbf_margin_normalized=scalar(),
            obstacle_clearance=scalar(),
            road_clearance=scalar(),
            kinematic_slack=scalar(),
            solver_finite=scalar(np.bool_),
            backup_feasible=scalar(np.bool_),
            recovery_success=scalar(np.bool_),
        )


# V2.3 deliberately uses a distinct replay tree.  Keeping the V2.2 tuple
# unchanged is important: old checkpoints and replay buffers must remain
# structurally reproducible and must never acquire new speed semantics by
# accident.  The common fields retain their exact order, followed by the four
# state-dependent speed-contract fields introduced by Guardian-DSAC.
GuardianDSACExperience = NamedTuple(
    "GuardianDSACExperience",
    [
        *GuardianCalQLExperience.__annotations__.items(),
        ("reachable_speed_lower", "jax.Array"),
        ("reachable_speed_upper", "jax.Array"),
        ("actual_vehicle_speed", "jax.Array"),
        ("emergency_stop", "jax.Array"),
    ],
)
GuardianDSACExperience.__doc__ = (
    "Guardian-DSAC V2.3 transition bound to target_speed_lateral_v1. "
    "Task Bellman updates consume action (the executed high-level command); "
    "raw_action remains diagnostic only."
)


def _guardian_dsac_batch_size(self) -> Optional[int]:
    return probe_batch_size(self.reward)


def _guardian_dsac_repr(self) -> str:
    return f"GuardianDSACExperience(size={self.batch_size()})"


def _guardian_dsac_create_example(
    obs_dim: int,
    action_dim: int,
    batch_size: Optional[int] = None,
):
    base = GuardianCalQLExperience.create_example(
        obs_dim,
        action_dim,
        batch_size=batch_size,
    )
    leading_dims = (batch_size,) if batch_size is not None else ()
    scalar = lambda dtype=np.float32: np.zeros(leading_dims, dtype=dtype)
    return GuardianDSACExperience(
        **base._asdict(),
        reachable_speed_lower=scalar(),
        reachable_speed_upper=scalar(),
        actual_vehicle_speed=scalar(),
        emergency_stop=scalar(np.bool_),
    )


GuardianDSACExperience.batch_size = _guardian_dsac_batch_size
GuardianDSACExperience.__repr__ = _guardian_dsac_repr
GuardianDSACExperience.create_example = staticmethod(_guardian_dsac_create_example)


# V2.4 owns a separate replay schema.  The stored Flow latent is part of the
# causal proposal record: it lets the Actor learn that the exact mode which
# triggered Guardian intervention should map to the state-aligned correction.
# Keeping these fields out of GuardianDSACExperience preserves every V2.3
# checkpoint and replay tree byte-for-byte.
GuardianFlowDSACExperience = NamedTuple(
    "GuardianFlowDSACExperience",
    [
        *GuardianDSACExperience.__annotations__.items(),
        ("flow_noise", "jax.Array"),
        ("flow_noise_valid", "jax.Array"),
    ],
)
GuardianFlowDSACExperience.__doc__ = (
    "Guardian-Flow-DSAC V2.4 transition. Task Bellman updates still consume "
    "only action (the executed command); flow_noise records the Gaussian "
    "latent that produced raw_action when flow_noise_valid is true."
)


def _guardian_flow_dsac_batch_size(self) -> Optional[int]:
    return probe_batch_size(self.reward)


def _guardian_flow_dsac_repr(self) -> str:
    return f"GuardianFlowDSACExperience(size={self.batch_size()})"


def _guardian_flow_dsac_create_example(
    obs_dim: int,
    action_dim: int,
    batch_size: Optional[int] = None,
):
    base = GuardianDSACExperience.create_example(
        obs_dim,
        action_dim,
        batch_size=batch_size,
    )
    leading_dims = (batch_size,) if batch_size is not None else ()
    return GuardianFlowDSACExperience(
        **base._asdict(),
        flow_noise=np.zeros((*leading_dims, action_dim), dtype=np.float32),
        flow_noise_valid=np.zeros(leading_dims, dtype=np.bool_),
    )


GuardianFlowDSACExperience.batch_size = _guardian_flow_dsac_batch_size
GuardianFlowDSACExperience.__repr__ = _guardian_flow_dsac_repr
GuardianFlowDSACExperience.create_example = staticmethod(
    _guardian_flow_dsac_create_example
)


# V2.4.1 keeps V2.4 replay immutable and adds an explicit episode-level
# latent-mode bit.  A valid all-zero Flow latent is therefore distinguishable
# from an offline row whose placeholder latent is zero but invalid.
GuardianFlowDSACLongRunExperience = NamedTuple(
    "GuardianFlowDSACLongRunExperience",
    [
        *GuardianFlowDSACExperience.__annotations__.items(),
        ("flow_zero_latent", "jax.Array"),
    ],
)
GuardianFlowDSACLongRunExperience.__doc__ = (
    "Guardian-Flow-DSAC V2.4.1 Long-Run transition. flow_zero_latent is true "
    "only for a causal deployment-branch proposal recorded with valid z=0."
)


def _guardian_flow_long_run_batch_size(self) -> Optional[int]:
    return probe_batch_size(self.reward)


def _guardian_flow_long_run_repr(self) -> str:
    return f"GuardianFlowDSACLongRunExperience(size={self.batch_size()})"


def _guardian_flow_long_run_create_example(
    obs_dim: int,
    action_dim: int,
    batch_size: Optional[int] = None,
):
    base = GuardianFlowDSACExperience.create_example(
        obs_dim,
        action_dim,
        batch_size=batch_size,
    )
    leading_dims = (batch_size,) if batch_size is not None else ()
    return GuardianFlowDSACLongRunExperience(
        **base._asdict(),
        flow_zero_latent=np.zeros(leading_dims, dtype=np.bool_),
    )


GuardianFlowDSACLongRunExperience.batch_size = (
    _guardian_flow_long_run_batch_size
)
GuardianFlowDSACLongRunExperience.__repr__ = _guardian_flow_long_run_repr
GuardianFlowDSACLongRunExperience.create_example = staticmethod(
    _guardian_flow_long_run_create_example
)


class MGFToyExperience(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    m_delta: "jax.Array"
    raw_action: "jax.Array"

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def __repr__(self):
        return f"MGFToyExperience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return MGFToyExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool_),
            m_delta=np.zeros(leading_dims, dtype=np.float32),
            raw_action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
        )

    @staticmethod
    def create(
        obs,
        action,
        reward,
        terminated,
        truncated,
        next_obs,
        info=None,
        *,
        store_executed_action: bool = False,
    ):
        info = info or {}
        raw_action = np.asarray(action, dtype=np.float32)
        executed = np.asarray(info.get("executed_target", action), dtype=np.float32)
        replay_action = executed if store_executed_action else raw_action
        return MGFToyExperience(
            obs=obs,
            action=replay_action,
            reward=reward,
            done=terminated or truncated,
            next_obs=next_obs,
            m_delta=np.asarray(info.get("m_delta", np.nan), dtype=np.float32),
            raw_action=raw_action,
        )


class MGFDriveExperience(NamedTuple):
    """Replay record for target-point drive MGF-SAC.

    `action` is always the exact certified action executed by the lower MPC;
    `raw_action` is retained only for projection-mismatch diagnostics.
    """

    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    m_delta: "jax.Array"
    raw_action: "jax.Array"
    projection_norm: "jax.Array"

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def __repr__(self):
        return f"MGFDriveExperience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return MGFDriveExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool_),
            m_delta=np.zeros(leading_dims, dtype=np.float32),
            raw_action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            projection_norm=np.zeros(leading_dims, dtype=np.float32),
        )

    @staticmethod
    def create(obs, action, reward, terminated, truncated, next_obs, info=None):
        info = info or {}
        mgf = info.get("mgf", {}) if isinstance(info, dict) else {}
        raw_action = np.asarray(
            mgf.get("raw_target", info.get("raw_target", action)), dtype=np.float32
        )
        executed_action = np.asarray(
            mgf.get("executed_target", info.get("executed_target", action)),
            dtype=np.float32,
        )
        return MGFDriveExperience(
            obs=obs,
            action=executed_action,
            reward=reward,
            done=terminated or truncated,
            next_obs=next_obs,
            m_delta=np.asarray(
                mgf.get("m_delta", info.get("m_delta", np.nan)), dtype=np.float32
            ),
            raw_action=raw_action,
            projection_norm=np.asarray(
                mgf.get("projection_norm", 0.0), dtype=np.float32
            ),
        )


class AEFLACTrajExperience(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    hj: "jax.Array"
    next_hj: "jax.Array"

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def __repr__(self):
        return f"AEFLACTrajExperience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        metric_shape = (*leading_dims, action_dim, action_dim)
        return AEFLACTrajExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool_),
            hj=np.zeros(metric_shape, dtype=np.float32),
            next_hj=np.zeros(metric_shape, dtype=np.float32),
        )

    @staticmethod
    def create(obs, action, reward, terminated, truncated, next_obs, info: dict, hj):
        ae_info = info.get("ae_flac", {}) if isinstance(info, dict) else {}
        next_hj = ae_info.get("next_hj")
        if next_hj is None:
            raise ValueError(
                "AEFLACTrajExperience.create requires info['ae_flac']['next_hj']"
            )
        return AEFLACTrajExperience(
            obs=obs,
            action=action,
            reward=reward,
            done=terminated,
            next_obs=next_obs,
            hj=hj,
            next_hj=next_hj,
        )


class GAEExperience(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    ret: "jax.Array"
    adv: "jax.Array"

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def __repr__(self):
        return f"GAEExperience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return GAEExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool_),
            ret=np.zeros(leading_dims, dtype=np.float32),
            adv=np.zeros(leading_dims, dtype=np.float32),
        )


class SafeExperience(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    cost: "jax.Array"
    feasible: "jax.Array"
    infeasible: "jax.Array"
    barrier: "jax.Array"
    next_barrier: "jax.Array"

    def batch_size(self) -> Optional[int]:
        try:
            if self.reward.ndim > 0:
                return self.reward.shape[0]
            else:
                return None
        except AttributeError:
            return None

    def __repr__(self):
        return f"SafeExperience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return SafeExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool_),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            cost=np.zeros(leading_dims, dtype=np.float32),
            feasible=np.zeros(leading_dims, dtype=np.bool_),
            infeasible=np.zeros(leading_dims, dtype=np.bool_),
            barrier=np.zeros(leading_dims, dtype=np.float32),
            next_barrier=np.zeros(leading_dims, dtype=np.float32),
        )

    @staticmethod
    def create(obs, action, reward, terminated, truncated, next_obs, info: dict):
        cost = info.get("cost", 0.0)
        feasible = info.get("feasible", False)
        infeasible = info.get("infeasible", False)
        barrier = info.get("barrier", 0.0)
        next_barrier = info.get("next_barrier", 0.0)
        return SafeExperience(
            obs=obs,
            action=action,
            reward=reward,
            done=terminated,
            next_obs=next_obs,
            cost=cost,
            feasible=feasible,
            infeasible=infeasible,
            barrier=barrier,
            next_barrier=next_barrier,
        )


class ObsActionPair(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return ObsActionPair(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
        )
