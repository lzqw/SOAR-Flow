"""Guardian-Flow-DSAC V2.4.2 dynamic skill consolidation.

V2.4.2 leaves the V2.4.1 Flow Actor, Q/P learners, Guardian and deployment
semantics unchanged.  It adds per-Seed deterministic Ordinary references and
strengthens only Seed9's existing Guardian-correction objective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm.guardian_dsac_flow_v2_4 import (
    _copy_tree,
    _python_metrics,
    _weighted_mean,
)
from relax.algorithm.guardian_dsac_flow_v2_4_1 import (
    GuardianDSACFlowV241,
    GuardianDSACFlowV241Config,
    _cosine_rate,
)
from relax.algorithm.guardian_dsac_v2_3 import (
    proxy_actor_objective,
    weighted_action_proximal,
)
from relax.utils.guardian_deployment_gate import ordinary_hard_safety_failures


@dataclass(frozen=True)
class GuardianDSACFlowV242Config(GuardianDSACFlowV241Config):
    """V2.4.2 loss binding; inherited fields remain V2.4.1-compatible."""

    skill_retention_weight: float = 0.10
    seed9_correction_weight: float = 4.0
    seed9_zero_latent_correction_weight: float = 2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        values = (
            self.skill_retention_weight,
            self.seed9_correction_weight,
            self.seed9_zero_latent_correction_weight,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("V2.4.2 consolidation weights must be finite/non-negative")


def _parameter_sha256(params) -> str:
    payload = pickle.dumps(jax.device_get(params), protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest()


def _same_tree_shape(left, right) -> bool:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    return bool(
        left_structure == right_structure
        and len(left_leaves) == len(right_leaves)
        and all(
            np.asarray(left_value).shape == np.asarray(right_value).shape
            for left_value, right_value in zip(left_leaves, right_leaves)
        )
    )


class GuardianDSACFlowV242(GuardianDSACFlowV241):
    """V2.4.1 plus dynamic per-Seed Ordinary skill retention."""

    FORMAT = "guardian_dsac_one_step_flow_v2_4_2_consolidation_v1"
    REPLAY_SCHEMA = "guardian_flow_dsac_v2_4_2_experience_v1"

    def __init__(
        self,
        base_algorithm,
        flow_policy,
        flow_params,
        *,
        config: GuardianDSACFlowV242Config = GuardianDSACFlowV242Config(),
    ):
        self.ordinary_safe_references: dict[int, dict[str, object]] = {}
        self.ordinary_safe_reference_history: list[dict[str, object]] = []
        self.skill_reference_cursor = 0
        super().__init__(base_algorithm, flow_policy, flow_params, config=config)
        self.config = config

    def _build_long_run_actor_update(self) -> None:
        config = self.config
        optimizer = self.online_actor_optimizer
        retention_seeds = tuple(int(seed) for seed in config.retention_teacher_seeds)
        correction_axis_weight = jnp.asarray(
            [config.correction_speed_weight, config.correction_lateral_weight],
            dtype=jnp.float32,
        )

        def online_actor_update(
            key,
            state,
            data,
            q1,
            q2,
            proxy,
            learning_rate,
            skill_observation,
            skill_target_action,
            skill_support,
        ):
            action_shape = jnp.asarray(data.action).shape
            (
                policy_key,
                diversity_1_key,
                diversity_2_key,
                retention_key,
            ) = jax.random.split(key, 4)
            noise = jax.random.normal(policy_key, action_shape, dtype=jnp.float32)
            diversity_noise_1 = jax.random.normal(
                diversity_1_key, action_shape, dtype=jnp.float32
            )
            diversity_noise_2 = jax.random.normal(
                diversity_2_key, action_shape, dtype=jnp.float32
            )
            retention_noise = jax.random.normal(
                retention_key, action_shape, dtype=jnp.float32
            )
            actor_support = jnp.asarray(data.actor_controlled, dtype=jnp.float32)
            proxy_support = jnp.maximum(
                jnp.asarray(data.guardian_dependence_target_valid, jnp.float32),
                jnp.asarray(data.guardian_dependence_corrected_valid, jnp.float32),
            )
            safe_action = jax.lax.stop_gradient(
                self.flow_policy.get_action(state.safe_one_step, data.obs, noise)
            )
            coefficient = jnp.maximum(
                config.proximal_weight_min,
                config.proximal_weight_initial
                * jnp.exp(
                    -jnp.asarray(state.proximal_step, dtype=jnp.float32)
                    / float(max(config.proximal_decay_steps, 1))
                ),
            )
            hard = jnp.asarray(data.intervention_hard, dtype=bool)
            soft = jnp.asarray(data.intervention_soft, dtype=bool)
            modified = jnp.asarray(data.actual_control_modified, dtype=bool)
            latent_valid = jnp.asarray(data.flow_noise_valid, dtype=bool)
            zero_latent_row = jnp.asarray(data.flow_zero_latent, dtype=bool)
            correction_support = modified & latent_valid & (hard | soft)
            correction_sample_weight = jnp.where(
                hard,
                1.0,
                jnp.where(soft, 0.5, 0.0),
            ).astype(jnp.float32) * correction_support.astype(jnp.float32)
            pair_valid = jnp.asarray(data.proxy_pair_valid, dtype=bool)
            correction_target = jnp.where(
                pair_valid[..., None],
                jnp.asarray(data.proxy_corrected_action, dtype=jnp.float32),
                jnp.asarray(data.action, dtype=jnp.float32),
            )
            stored_noise = jnp.asarray(data.flow_noise, dtype=jnp.float32)
            scene_seed = jnp.asarray(data.scene_seed, dtype=jnp.int32)
            seed9 = scene_seed == 9
            correction_scale = jnp.where(
                seed9,
                config.seed9_correction_weight,
                config.correction_weight,
            ).astype(jnp.float32)
            zero_correction_scale = jnp.where(
                seed9,
                config.seed9_zero_latent_correction_weight,
                config.zero_latent_correction_weight,
            ).astype(jnp.float32)
            retention_seed = jnp.zeros_like(scene_seed, dtype=bool)
            for seed in retention_seeds:
                retention_seed = retention_seed | (scene_seed == seed)
            retention_support = (~latent_valid) & retention_seed
            retention_weight = retention_support.astype(jnp.float32)
            retention_teacher = jax.lax.stop_gradient(
                self.flow_policy.bc_action(
                    state.params.bc_flow,
                    data.obs,
                    retention_noise,
                )
            )
            skill_observation = jnp.asarray(skill_observation, dtype=jnp.float32)
            skill_target_action = jax.lax.stop_gradient(
                jnp.asarray(skill_target_action, dtype=jnp.float32)
            )
            skill_support = jnp.asarray(skill_support, dtype=jnp.float32)

            def loss_fn(params):
                action = self.flow_policy.get_action(params, data.obs, noise)
                mean_1, _ = self.base.agent.q_distribution(q1, data.obs, action)
                mean_2, _ = self.base.agent.q_distribution(q2, data.obs, action)
                q_loss = -_weighted_mean(
                    jnp.minimum(mean_1, mean_2), actor_support
                )
                proxy_logit = self.base.agent.proxy_value(proxy, data.obs, action)
                p_probability = jax.nn.sigmoid(proxy_logit)
                p_loss = _weighted_mean(
                    proxy_actor_objective(
                        proxy_logit,
                        objective=self.base.config.proxy_actor_objective,
                    ),
                    proxy_support,
                )
                delta = action - safe_action
                proximal, speed_drift, lateral_drift = weighted_action_proximal(
                    delta,
                    speed_weight=config.proximal_speed_weight,
                    lateral_weight=config.proximal_lateral_weight,
                )
                stored_prediction = self.flow_policy.get_action(
                    params, data.obs, stored_noise
                )
                zero_prediction = self.flow_policy.get_action(
                    params,
                    data.obs,
                    jnp.zeros_like(stored_noise),
                )
                target = jax.lax.stop_gradient(correction_target)
                stored_square = jnp.square(stored_prediction - target)
                zero_square = jnp.square(zero_prediction - target)
                stored_error = jnp.sum(
                    stored_square * correction_axis_weight, axis=-1
                )
                zero_error = jnp.sum(zero_square * correction_axis_weight, axis=-1)
                stored_correction = _weighted_mean(
                    stored_error, correction_sample_weight
                )
                zero_correction = _weighted_mean(
                    zero_error, correction_sample_weight
                )
                correction = _weighted_mean(
                    correction_scale
                    * (stored_error + zero_correction_scale * zero_error),
                    correction_sample_weight,
                )
                retention_action = self.flow_policy.get_action(
                    params, data.obs, retention_noise
                )
                retention_square = jnp.sum(
                    jnp.square(retention_action - retention_teacher), axis=-1
                )
                retention = _weighted_mean(retention_square, retention_weight)
                skill_prediction = self.flow_policy.get_action(
                    params,
                    skill_observation,
                    jnp.zeros_like(skill_target_action),
                )
                skill_square = jnp.sum(
                    jnp.square(skill_prediction - skill_target_action), axis=-1
                )
                skill = _weighted_mean(skill_square, skill_support)
                diversity_action_1 = self.flow_policy.get_action(
                    params, data.obs, diversity_noise_1
                )
                diversity_action_2 = self.flow_policy.get_action(
                    params, data.obs, diversity_noise_2
                )
                diversity = jnp.mean(
                    jnp.linalg.norm(
                        diversity_action_1 - diversity_action_2,
                        axis=-1,
                    )
                )
                total = (
                    config.q_actor_weight * q_loss
                    + config.proxy_actor_weight * p_loss
                    + coefficient * proximal
                    + correction
                    + config.retention_teacher_weight * retention
                    + config.skill_retention_weight * skill
                )
                return total, (
                    q_loss,
                    p_loss,
                    p_probability,
                    proximal,
                    speed_drift,
                    lateral_drift,
                    correction,
                    stored_correction,
                    zero_correction,
                    stored_square,
                    zero_square,
                    retention,
                    skill,
                    diversity,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params.one_step
            )
            adam_updates, opt_state = optimizer.update(
                grads,
                state.actor_opt_state,
                state.params.one_step,
            )
            updates = jax.tree_util.tree_map(
                lambda value: -jnp.asarray(learning_rate, jnp.float32) * value,
                adam_updates,
            )
            params = optax.apply_updates(state.params.one_step, updates)
            target = optax.incremental_update(
                params, state.target_one_step, config.target_actor_tau
            )
            (
                q_loss,
                p_loss,
                p_probability,
                proximal,
                speed_drift,
                lateral_drift,
                correction,
                stored_correction,
                zero_correction,
                stored_square,
                zero_square,
                retention,
                skill,
                diversity,
            ) = aux
            correction_weight_mean = _weighted_mean(
                correction_scale, correction_sample_weight
            )
            zero_coefficient_mean = _weighted_mean(
                zero_correction_scale, correction_sample_weight
            )
            return state._replace(
                params=state.params._replace(one_step=params),
                actor_opt_state=opt_state,
                target_one_step=target,
                gradient_step=state.gradient_step + 1,
                proximal_step=state.proximal_step + 1,
            ), {
                "flow_actor_loss": loss,
                "flow_q_loss": q_loss,
                "flow_proxy_loss": p_loss,
                "flow_absolute_proxy_loss": _weighted_mean(
                    p_probability, proxy_support
                ),
                "flow_proximal_loss": proximal,
                "flow_proximal_speed_drift": speed_drift,
                "flow_proximal_lateral_drift": lateral_drift,
                "flow_proximal_coefficient": coefficient,
                "flow_correction_loss": correction,
                "flow_correction_stored_loss": stored_correction,
                "flow_correction_zero_latent_loss": zero_correction,
                "flow_correction_unweighted_loss": (
                    stored_correction + zero_correction
                ),
                "flow_correction_effective_weight_mean": correction_weight_mean,
                "flow_zero_correction_coefficient_mean": zero_coefficient_mean,
                "flow_correction_support_fraction": jnp.mean(
                    correction_support.astype(jnp.float32)
                ),
                "flow_correction_seed9_support_fraction": jnp.mean(
                    (correction_support & seed9).astype(jnp.float32)
                ),
                "flow_correction_weight_mean": jnp.mean(correction_sample_weight),
                "flow_correction_error_speed": _weighted_mean(
                    stored_square[..., 0], correction_sample_weight
                ),
                "flow_correction_error_lateral": _weighted_mean(
                    stored_square[..., 1], correction_sample_weight
                ),
                "flow_zero_correction_error_speed": _weighted_mean(
                    zero_square[..., 0], correction_sample_weight
                ),
                "flow_zero_correction_error_lateral": _weighted_mean(
                    zero_square[..., 1], correction_sample_weight
                ),
                "flow_correction_proxy_target_fraction": _weighted_mean(
                    pair_valid.astype(jnp.float32), correction_sample_weight
                ),
                "flow_retention_loss": retention,
                "flow_retention_support_fraction": jnp.mean(retention_weight),
                "flow_skill_retention_loss": skill,
                "flow_skill_retention_support_fraction": jnp.mean(skill_support),
                "flow_zero_latent_batch_fraction": jnp.mean(
                    (zero_latent_row & latent_valid).astype(jnp.float32)
                ),
                "flow_latent_action_diversity": diversity,
                "flow_actor_learning_rate": learning_rate,
            }

        self._online_actor_update_long_run = jax.jit(online_actor_update)

    def online_joint_update(
        self,
        key,
        data,
        *,
        actor_data=None,
        update_actor: bool = True,
        skill_data=None,
        skill_reference_seed: int | None = None,
    ):
        if not self.bc_flow_frozen:
            raise RuntimeError("online training requires a frozen Flow teacher")
        if not self.online_schedule_prepared:
            raise RuntimeError("prepare V2.4.2 online schedules before training")
        critic_key, next_key, current_key, actor_key = jax.random.split(key, 4)
        next_noise = jax.random.normal(
            next_key, jnp.asarray(data.action).shape, dtype=jnp.float32
        )
        current_noise = jax.random.normal(
            current_key, jnp.asarray(data.action).shape, dtype=jnp.float32
        )
        next_action = self.flow_policy.get_action(
            self.state.target_one_step, data.next_obs, next_noise
        )
        policy_action = self.flow_policy.get_action(
            self.state.params.one_step, data.obs, current_noise
        )
        critic_lr = float(
            np.asarray(self._critic_learning_rate(self.online_critic_updates_total))
        )
        proxy_lr = float(
            np.asarray(self._proxy_learning_rate(self.online_critic_updates_total))
        )
        q_metrics = self.base.online_critic_update_with_policy_actions(
            critic_key,
            data,
            next_action=next_action,
            policy_action=policy_action,
        )
        p_metrics = self.base.proxy_update(data)
        self.online_critic_updates_total += 1
        metrics = {
            **{f"task_{name}": value for name, value in q_metrics.items()},
            **p_metrics,
            "flow_implicit_entropy_target": 0.0,
            "flow_actor_entropy_free": 1.0,
            "flow_bc_frozen": 1.0,
            "task_critic_learning_rate": critic_lr,
            "proxy_learning_rate": proxy_lr,
        }
        if update_actor:
            actor_batch = data if actor_data is None else actor_data
            dummy_count = min(128, int(np.asarray(actor_batch.obs).shape[0]))
            skill_observation = jnp.asarray(actor_batch.obs[:dummy_count])
            skill_target = jnp.zeros(
                (dummy_count, int(np.asarray(actor_batch.action).shape[-1])),
                dtype=jnp.float32,
            )
            skill_support = jnp.zeros((dummy_count,), dtype=jnp.float32)
            used_reference_seed = -1
            if skill_reference_seed is not None:
                seed = int(skill_reference_seed)
                if seed not in self.ordinary_safe_references:
                    raise ValueError(f"Seed{seed} has no Ordinary-safe reference")
                if skill_data is None:
                    raise ValueError("skill replay data is required for an active reference")
                observed_seeds = np.asarray(skill_data.scene_seed, dtype=np.int32)
                if observed_seeds.ndim != 1 or not np.all(observed_seeds == seed):
                    raise ValueError("skill replay data must contain only the reference Seed")
                skill_observation = jnp.asarray(skill_data.obs, dtype=jnp.float32)
                zeros = jnp.zeros_like(jnp.asarray(skill_data.action, dtype=jnp.float32))
                skill_target = jax.lax.stop_gradient(
                    self.flow_policy.get_action(
                        self.ordinary_safe_references[seed]["params"],
                        skill_observation,
                        zeros,
                    )
                )
                skill_support = jnp.ones(observed_seeds.shape, dtype=jnp.float32)
                used_reference_seed = seed
            actor_lr = _cosine_rate(
                self.config.actor_learning_rate_initial,
                self.config.actor_learning_rate_final,
                self.online_actor_updates_total,
                self.config.actor_learning_rate_decay_steps,
            )
            self.state, actor_metrics = self._online_actor_update_long_run(
                actor_key,
                self.state,
                actor_batch,
                self.base.state.params.q1,
                self.base.state.params.q2,
                self.base.state.params.proxy,
                jnp.asarray(actor_lr, dtype=jnp.float32),
                skill_observation,
                skill_target,
                skill_support,
            )
            self.online_actor_updates_total += 1
            metrics.update(_python_metrics(actor_metrics))
            metrics["flow_skill_reference_seed"] = used_reference_seed
        return metrics

    def consider_ordinary_safe_references(
        self,
        evaluation,
        *,
        step: int,
        label: str,
    ) -> list[dict[str, object]]:
        """Freeze first-safe Actors and replace them only on higher safe return."""

        per_seed = evaluation.get("per_seed", {})
        actor_hash = _parameter_sha256(self.state.params.one_step)
        events: list[dict[str, object]] = []
        for seed_text in sorted(per_seed, key=lambda value: int(value)):
            seed = int(seed_text)
            if seed < 0 or seed > 9:
                raise ValueError(f"unbound Ordinary reference Seed {seed}")
            ordinary = per_seed[seed_text]["ordinary"]
            failures = ordinary_hard_safety_failures(ordinary)
            ordinary_return = float(ordinary["mean_return"])
            if not np.isfinite(ordinary_return):
                failures = sorted(set([*failures, "non_finite_return"]))
            previous = self.ordinary_safe_references.get(seed)
            decision = "unsafe"
            if not failures:
                if previous is None:
                    decision = "first_ordinary_safe"
                elif ordinary_return > float(previous["ordinary_return"]):
                    decision = "higher_return_ordinary_safe"
                else:
                    decision = "kept_higher_return_reference"
                if decision != "kept_higher_return_reference":
                    params = _copy_tree(self.state.params.one_step)
                    self.ordinary_safe_references[seed] = {
                        "params": params,
                        "ordinary_return": ordinary_return,
                        "step": int(step),
                        "label": str(label),
                        "parameter_sha256": _parameter_sha256(params),
                    }
            selected = self.ordinary_safe_references.get(seed)
            event = {
                "seed": seed,
                "step": int(step),
                "label": str(label),
                "ordinary_safe": not failures,
                "ordinary_failures": failures,
                "ordinary_return": ordinary_return,
                "previous_reference_return": (
                    None if previous is None else float(previous["ordinary_return"])
                ),
                "decision": decision,
                "evaluated_actor_parameter_sha256": actor_hash,
                "selected_reference_parameter_sha256": (
                    None if selected is None else selected["parameter_sha256"]
                ),
                "selected_reference_return": (
                    None if selected is None else float(selected["ordinary_return"])
                ),
            }
            events.append(event)
        self.ordinary_safe_reference_history.extend(events)
        return events

    def select_skill_reference_seed(self, available_seeds) -> int | None:
        available = tuple(
            seed
            for seed in sorted(int(value) for value in available_seeds)
            if seed in self.ordinary_safe_references
        )
        if not available:
            return None
        seed = available[self.skill_reference_cursor % len(available)]
        self.skill_reference_cursor += 1
        return seed

    def ordinary_safe_reference_metadata(self) -> dict[str, dict[str, object]]:
        return {
            str(seed): {
                key: value
                for key, value in reference.items()
                if key != "params"
            }
            for seed, reference in sorted(self.ordinary_safe_references.items())
        }

    def save_ordinary_safe_references(self, path: str | Path) -> None:
        payload = {
            "format": "guardian_flow_dsac_v2_4_2_ordinary_safe_references_v1",
            "references": jax.device_get(self.ordinary_safe_references),
            "history": list(self.ordinary_safe_reference_history),
            "skill_reference_cursor": int(self.skill_reference_cursor),
        }
        with Path(path).open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def _checkpoint_payload(self):
        payload = super()._checkpoint_payload()
        payload.update(
            format=self.FORMAT,
            flow_config=asdict(self.config),
            replay_schema=self.REPLAY_SCHEMA,
            ordinary_safe_reference_by_seed=jax.device_get(
                self.ordinary_safe_references
            ),
            ordinary_safe_reference_history=list(
                self.ordinary_safe_reference_history
            ),
            skill_reference_cursor=int(self.skill_reference_cursor),
            dynamic_ordinary_safe_retention=True,
            seed9_weighted_correction=True,
        )
        return payload

    def _load_checkpoint_payload(self, payload, *, load_base: bool = True):
        required = {
            "ordinary_safe_reference_by_seed",
            "ordinary_safe_reference_history",
            "skill_reference_cursor",
            "dynamic_ordinary_safe_retention",
            "seed9_weighted_correction",
        }
        if not isinstance(payload, dict) or payload.get("format") != self.FORMAT:
            raise ValueError("checkpoint is not Guardian-Flow-DSAC V2.4.2")
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"V2.4.2 checkpoint missing fields: {missing}")
        if payload["dynamic_ordinary_safe_retention"] is not True:
            raise ValueError("V2.4.2 checkpoint disables dynamic skill retention")
        if payload["seed9_weighted_correction"] is not True:
            raise ValueError("V2.4.2 checkpoint disables Seed9 correction weighting")
        super()._load_checkpoint_payload(payload, load_base=load_base)
        references = {}
        for seed_text, value in payload["ordinary_safe_reference_by_seed"].items():
            seed = int(seed_text)
            if seed < 0 or seed > 9:
                raise ValueError("V2.4.2 checkpoint contains an unbound reference Seed")
            ordinary_return = float(value["ordinary_return"])
            step = int(value["step"])
            params = jax.device_put(value["params"])
            if not np.isfinite(ordinary_return) or step < 0:
                raise ValueError("V2.4.2 checkpoint contains invalid reference metadata")
            if not _same_tree_shape(params, self.state.params.one_step):
                raise ValueError("V2.4.2 reference Actor architecture mismatch")
            expected_hash = str(value["parameter_sha256"])
            if _parameter_sha256(params) != expected_hash:
                raise ValueError("V2.4.2 reference Actor hash mismatch")
            references[seed] = {
                **value,
                "params": params,
                "ordinary_return": ordinary_return,
                "step": step,
                "parameter_sha256": expected_hash,
            }
        cursor = int(payload["skill_reference_cursor"])
        if cursor < 0:
            raise ValueError("V2.4.2 checkpoint has a negative reference cursor")
        self.ordinary_safe_references = references
        self.ordinary_safe_reference_history = list(
            payload["ordinary_safe_reference_history"]
        )
        self.skill_reference_cursor = cursor


__all__ = [
    "GuardianDSACFlowV242",
    "GuardianDSACFlowV242Config",
]
