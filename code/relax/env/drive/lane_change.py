import logging, importlib, inspect
import sys
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from gymnasium.spaces.utils import FlatType
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod, FirstPGBlock

# ===== 可调参数（如需更远就把 REL 调大一些）=====
OBSTACLE_OFFSET_REL = 1.5       # 相对距离倍数：s = s0 + REL*L + M
OBSTACLE_OFFSET_M   = 0.0       # 绝对偏移（米，正向前、负向后）
SAFETY_MARGIN_END   = 2.0       # 距离段尾安全余量（米）
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from relax.env.registration_utils import safe_register
# 注册你的新环境
sys.path.append(".")  # 确保当前目录在路径中，以便找到 lane_change 模块
safe_register(
    id='FlatThreeLaneStraight',  # <-- 这是你将在训练脚本中使用的新 ID
    entry_point='relax.env.drive.lane_change:make_flat_metadrive_env',  # 指向你的创建函数
    max_episode_steps=1000  # 你可以根据需要调整这个值
)

safe_register(
    id='FlatThreeLaneStraightMultiObs',
    entry_point='relax.env.drive.lane_change:make_flat_metadrive_multi_obs_env',
    max_episode_steps=1000
)

def make_flat_metadrive_env():
    """
    这是一个辅助函数，用于创建 MetaDrive 环境
    并应用 FlattenObservation 包装器。
    """
    # 你可以在这里传递配置，例如禁用渲染以加快训练速度
    config = {
        "use_render": False,
        "log_level": 50  # 设置为 ERROR，减少不必要的日志输出
    }
    env = ThreeLaneStraightEnv(config)

    # 关键步骤：将 Dict 观测空间 扁平化为 Box 空间
    env = FlattenObservation(env)
    return env


def make_flat_metadrive_multi_obs_env():
    config = {
        "use_render": False,
        "log_level": 50
    }
    env = ThreeLaneStraightMultiObsEnv(config)
    env = FlattenObservation(env)
    return env


def _pick_vehicle_class() -> Tuple[type, str]:
    """
    优先：TrafficVehicle -> IDMVehicle -> 其他非 DefaultVehicle 子类 -> DefaultVehicle(兜底)
    绝不返回抽象 BaseVehicle。
    """
    try:
        from metadrive.component.vehicle.vehicle_type import TrafficVehicle
        return TrafficVehicle, "TrafficVehicle"
    except Exception:
        pass
    try:
        from metadrive.component.vehicle.vehicle_type import IDMVehicle
        return IDMVehicle, "IDMVehicle"
    except Exception:
        pass
    # 其他子类（排除 DefaultVehicle 与抽象 BaseVehicle）
    try:
        vt = importlib.import_module("metadrive.component.vehicle.vehicle_type")
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        for name in dir(vt):
            obj = getattr(vt, name)
            if inspect.isclass(obj):
                try:
                    if issubclass(obj, BaseVehicle) and obj is not BaseVehicle and name.lower() != "defaultvehicle":
                        return obj, name
                except Exception:
                    continue
    except Exception:
        pass
    # 兜底
    from metadrive.component.vehicle.vehicle_type import DefaultVehicle
    return DefaultVehicle, "DefaultVehicle(FALLBACK)"


def build_obstacle_circle_approx(
    center_xy: np.ndarray,
    heading: float,
    length: float,
    width: float,
    max_circles: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    center_xy = np.asarray(center_xy, dtype=np.float32).reshape(2)
    length = float(length)
    width = float(width)
    max_circles = max(int(max_circles), 1)

    circle_radius = max(width * 0.5, 1e-3)
    nominal_count = int(np.ceil(length / max(2.0 * circle_radius, 1e-3)))
    n_circles = min(max(nominal_count, 1), max_circles)

    spacing = length / n_circles
    offsets = np.linspace(
        -0.5 * length + 0.5 * spacing,
        0.5 * length - 0.5 * spacing,
        n_circles,
        dtype=np.float32,
    )
    direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
    positions = center_xy[None, :] + offsets[:, None] * direction[None, :]
    radii = np.full((n_circles,), circle_radius, dtype=np.float32)
    return positions.astype(np.float32), radii


def refresh_post_spawn_observation(
    observation,
    observation_handlers,
    agents,
):
    """Re-observe after custom obstacles are spawned, without advancing time.

    MetaDrive computes the observation returned by ``super().reset()`` before
    this environment creates its explicit parked vehicles.  Calling the
    already-constructed observation handler again makes the first policy
    input reflect those vehicles while avoiding a hidden dummy control step.
    The fallback keeps compatibility with older/mocked MetaDrive versions.
    """

    if not isinstance(observation_handlers, dict) or not observation_handlers:
        return observation, False
    if not isinstance(agents, dict) or not agents:
        return observation, False
    try:
        if isinstance(observation, dict):
            refreshed = dict(observation)
            changed = False
            for agent_id, handler in observation_handlers.items():
                vehicle = agents.get(agent_id)
                if vehicle is None or agent_id not in refreshed:
                    continue
                refreshed[agent_id] = handler.observe(vehicle)
                changed = True
            return refreshed, changed

        if len(observation_handlers) != 1:
            return observation, False
        agent_id, handler = next(iter(observation_handlers.items()))
        vehicle = agents.get(agent_id)
        if vehicle is None and len(agents) == 1:
            vehicle = next(iter(agents.values()))
        if vehicle is None:
            return observation, False
        return handler.observe(vehicle), True
    except Exception:
        return observation, False


@dataclass(frozen=True)
class StaticObstacleSpec:
    lane_index: int
    spawn_mode: str = "fraction"
    longitudinal_fraction: float = 0.5
    offset_rel: float = OBSTACLE_OFFSET_REL
    offset_m: float = OBSTACLE_OFFSET_M
    length: float = 12.0
    width: float = 2.5
    vehicle_model: str = "bus"
    circle_count: int = 4
    circle_radius: float | None = None


class ThreeLaneStraightEnv(SafeMetaDriveEnv):
    """官方 SafeMetaDriveEnv 上的固定三车道、多静态障碍安全场景。"""

    def default_config(self):
        cfg = super().default_config()

        # —— 固定为我们自定义的直线地图 ——
        cfg["map"] = None
        cfg["map_config"].update({
            BaseMap.GENERATE_TYPE:   MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            BaseMap.GENERATE_CONFIG: "S",   # 只有一个直线区块
            BaseMap.LANE_NUM:        3,
            BaseMap.LANE_WIDTH:      3.7,
        })

        # —— 固定出生在 S 段中间车道起点附近 ——
        cfg["random_spawn_lane_index"] = False
        cfg["agent_configs"]["default_agent"].update({
            "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 1),  # 中间车道
            "spawn_longitude":  1.0,   # s0：离段起点 1m
            "spawn_lateral":    0.0,
        })

        # 其他参数（如 manual_control/use_render 等）由外部传入 cfg 控制，这里不强制覆盖
        # 随机交通关闭；场景中的显式静态障碍由 reset 参数统一生成。
        cfg.update(dict(
            start_seed=0,
            log_level=logging.ERROR,
            num_scenarios=1,
            # Keep the official SafeMetaDriveEnv cost channel independent
            # from reward. Candidate deployment gates consume cost/total_cost
            # directly, while CBF/REC remains the online shield.
            cost_to_reward=False,
            crash_vehicle_done=False,
            crash_object_done=False,
            out_of_road_done=True,
            traffic_density=0.0,
            random_traffic=False,
            accident_prob=0.0,
            traffic_mode="NoTraffic",
            # 关闭 MetaDrive 原生 lateral_reward，改由 MPCControlWrapper 统一计算（方便调参）
            use_lateral_reward=False,
        ))
        return cfg

    def _clear_obstacle_cache(self):
        self.obstacle_world_positions = np.zeros((0, 2), dtype=np.float32)
        self.obstacle_radii = np.zeros((0,), dtype=np.float32)

    def get_obstacle_info(self):
        if not hasattr(self, "obstacle_world_positions") or not hasattr(self, "obstacle_radii"):
            self._clear_obstacle_cache()
        return self.obstacle_world_positions.copy(), self.obstacle_radii.copy()

    def _get_reset_seed(self, seed):
        return 0

    def _clear_spawned_parked_objects(self):
        try:
            eng = getattr(self, "engine", None)
            parked_objects = []
            if getattr(self, "_parked_obj", None) is not None:
                parked_objects.append(self._parked_obj)
            parked_objects.extend(getattr(self, "_parked_objs", []))

            if eng is not None:
                ids = [obj.id for obj in parked_objects if obj is not None and hasattr(obj, "id")]
                if ids:
                    try:
                        eng.clear_objects(ids)
                    except Exception:
                        pass
                    try:
                        if hasattr(eng, "before_step"):
                            eng.before_step({})
                    except Exception:
                        pass
            self._parked_obj = None
            self._parked_objs = []
        except Exception:
            pass

    def _build_static_obstacle_specs(self, lane_length: float, spawn_longitude: float):
        return [
            StaticObstacleSpec(
                lane_index=1,
                spawn_mode="legacy",
                offset_rel=OBSTACLE_OFFSET_REL,
                offset_m=OBSTACLE_OFFSET_M,
                length=12.0,
                width=2.5,
                vehicle_model="bus",
                circle_count=4,
            )
        ]

    def _resolve_obstacle_longitude(
        self,
        spec: StaticObstacleSpec,
        lane_length: float,
        spawn_longitude: float,
    ) -> float:
        if spec.spawn_mode == "legacy":
            s_raw = spawn_longitude + spec.offset_rel * lane_length + spec.offset_m
        else:
            usable_length = max(lane_length - spawn_longitude - SAFETY_MARGIN_END, 0.0)
            s_raw = spawn_longitude + spec.longitudinal_fraction * usable_length + spec.offset_m
        return max(0.0, min(s_raw, max(lane_length - SAFETY_MARGIN_END, 0.0)))

    def _build_obstacle_marker_approx(
        self,
        center_xy: np.ndarray,
        heading: float,
        spec: StaticObstacleSpec,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if spec.circle_count <= 1:
            radius = spec.circle_radius
            if radius is None:
                radius = max(spec.width * 0.5, 1e-3)
            positions = np.asarray(center_xy, dtype=np.float32).reshape(1, 2)
            radii = np.asarray([radius], dtype=np.float32)
            return positions, radii

        positions, radii = build_obstacle_circle_approx(
            center_xy=center_xy,
            heading=heading,
            length=spec.length,
            width=spec.width,
            max_circles=spec.circle_count,
        )
        if spec.circle_radius is not None:
            radii = np.full_like(radii, spec.circle_radius, dtype=np.float32)
        return positions, radii

    def _spawn_static_obstacles(self, lane_length: float, spawn_longitude: float):
        specs = self._build_static_obstacle_specs(lane_length, spawn_longitude)
        obstacle_positions = []
        obstacle_radii = []
        parked_objects = []
        debug_rows = []

        VehCls, vname = _pick_vehicle_class()
        for idx, spec in enumerate(specs):
            lane = self.current_map.road_network.get_lane(
                (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, int(spec.lane_index))
            )
            s_target = self._resolve_obstacle_longitude(spec, lane_length, spawn_longitude)
            x, y = lane.position(s_target, 0.0)
            heading = float(lane.heading_theta_at(s_target))

            v_cfg = {
                "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, int(spec.lane_index)),
                "spawn_longitude": float(s_target),
                "spawn_lateral": 0.0,
                "spawn_velocity": [0.0, 0.0],
                "spawn_velocity_car_frame": True,
                "spawn_position_heading": ([float(x), float(y)], heading),
                "enable_reverse": False,
                "max_speed_km_h": 0.0,
                "random_color": False,
                "vehicle_model": spec.vehicle_model,
                "length": float(spec.length),
                "width": float(spec.width),
            }

            parked = self.engine.spawn_object(VehCls, vehicle_config=v_cfg)
            parked_objects.append(parked)
            positions, radii = self._build_obstacle_marker_approx(
                center_xy=np.array([float(x), float(y)], dtype=np.float32),
                heading=heading,
                spec=spec,
            )
            obstacle_positions.append(positions)
            obstacle_radii.append(radii)

            try:
                if hasattr(parked, "set_break_down"):
                    parked.set_break_down(True)
                if hasattr(parked, "policy"):
                    parked.policy = None
                if hasattr(parked, "controller"):
                    parked.controller = None
                if hasattr(parked, "enable_navigation"):
                    parked.enable_navigation(False)
                if hasattr(parked, "set_angular_velocity"):
                    parked.set_angular_velocity([0.0, 0.0, 0.0])
            except Exception:
                pass

            debug_rows.append(
                f"#{idx}:lane={spec.lane_index},s={s_target:.2f},pos=({x:.2f},{y:.2f}),circles={positions.shape[0]}"
            )

        self._parked_objs = parked_objects
        self._parked_obj = parked_objects[0] if parked_objects else None
        if obstacle_positions:
            self.obstacle_world_positions = np.concatenate(obstacle_positions, axis=0).astype(np.float32)
            self.obstacle_radii = np.concatenate(obstacle_radii, axis=0).astype(np.float32)
        else:
            self._clear_obstacle_cache()

        if not hasattr(self, "_printed"):
            try:
                print(
                    f"[{self.__class__.__name__}] veh_cls={vname}, obstacles={len(specs)}, "
                    + " | ".join(debug_rows)
                )
            except Exception:
                pass
            self._printed = True

    # 兼容 Gym / Gymnasium 的 reset（MetaDrive 0.4.3 无 options 参数）
    def reset(self, seed=None, options=None):
        self._clear_obstacle_cache()
        self._clear_spawned_parked_objects()
        seed = self._get_reset_seed(seed)

        # 不传 options，以兼容 0.4.3 的 BaseEnv.reset 签名
        out = super().reset(seed=seed) if seed is not None else super().reset()
        obs, info = (out if (isinstance(out, tuple) and len(out) == 2) else (out, {}))

        # ===== 计算 S 段（NODE_2 -> NODE_3）中间车道上的目标点 =====
        lane_S = self.current_map.road_network.get_lane((FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 1))

        # 段总长度 L
        L = getattr(lane_S, "length", None)
        if L is None:
            L = lane_S.get_length() if hasattr(lane_S, "get_length") else 0.0
        L = float(L)

        # 出生点 s0
        s0 = float(self.config["agent_configs"]["default_agent"]["spawn_longitude"])

        self._spawn_static_obstacles(L, s0)

        # ``super().reset()`` observed the world before the parked vehicles
        # above existed. Refresh the native sensor state without moving the
        # ego vehicle or consuming a reward/episode step.
        obs, observation_refreshed = refresh_post_spawn_observation(
            obs,
            getattr(self, "observations", None),
            getattr(self, "agents", None),
        )

        info = dict(info)
        info["safe_env_backend"] = (
            "metadrive.envs.safe_metadrive_env.SafeMetaDriveEnv"
        )
        info["safe_env_cost_enabled"] = True
        info["post_spawn_observation_refreshed"] = bool(
            observation_refreshed
        )

        return obs, info


class ThreeLaneStraightMultiObsEnv(ThreeLaneStraightEnv):
    """Randomized straight-road slalom with multiple parked vehicles."""

    def default_config(self):
        cfg = super().default_config()
        # MetaDrive interprets reset(seed=...) as a scenario index. MPC workers
        # use large 32-bit seeds, so we expose a wider valid scenario range here.
        cfg["num_scenarios"] = 100000
        return cfg

    def _get_reset_seed(self, seed):
        if seed is None:
            return None
        start_seed = int(self.config.get("start_seed", 0))
        num_scenarios = max(int(self.config.get("num_scenarios", 1)), 1)
        return start_seed + (int(seed) - start_seed) % num_scenarios

    def _build_static_obstacle_specs(self, lane_length: float, spawn_longitude: float):
        del lane_length, spawn_longitude
        rng = getattr(self, "np_random", None)
        mid_fraction = 0.36
        second_fraction = 0.58
        third_fraction = 0.78
        side_lanes = [0, 2]

        if rng is not None:
            mid_fraction = float(rng.uniform(0.30, 0.40))
            second_fraction = float(rng.uniform(0.52, 0.64))
            min_third_fraction = min(max(second_fraction + 0.14, 0.72), 0.84)
            third_fraction = float(rng.uniform(min_third_fraction, 0.90))
            if float(rng.uniform(0.0, 1.0)) < 0.5:
                side_lanes = [2, 0]

        return [
            StaticObstacleSpec(
                lane_index=1,
                longitudinal_fraction=mid_fraction,
                length=8.5,
                width=2.6,
                vehicle_model="bus",
                circle_count=1,
                circle_radius=1.75,
            ),
            StaticObstacleSpec(
                lane_index=side_lanes[0],
                longitudinal_fraction=second_fraction,
                length=8.0,
                width=2.5,
                vehicle_model="bus",
                circle_count=1,
                circle_radius=1.65,
            ),
            StaticObstacleSpec(
                lane_index=side_lanes[1],
                longitudinal_fraction=third_fraction,
                length=8.0,
                width=2.5,
                vehicle_model="bus",
                circle_count=1,
                circle_radius=1.65,
            ),
        ]
