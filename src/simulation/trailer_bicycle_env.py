from __future__ import annotations
from typing import Any

from src.simulation.config.trailer_bicycle_config import (
    TrackConfig,
    VehicleConfig,
    SimulationConfig,
    TrailerBicycleEnvConfig,
)
from src.utils.track import TrackModel, TrackProjection
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pandas as pd


from src.simulation.rendering import PyBulletMirrorRenderer


from dataclasses import dataclass, replace


def wrap_angle(angle: float) -> float:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


@dataclass(slots=True)
class VehicleState:
    x: float
    y: float
    yaw_truck: float
    yaw_trailer: float
    vx: float  # Truck frame
    vy: float  # Truck frame
    yaw_truck_rate: float
    yaw_trailer_rate: float
    steer: float
    accel: float


def compute_fy(alpha, cc, fz, fx, mu, gamma):

    fy_max = jnp.sqrt((mu * fz) ** 2 - gamma * fx**2)

    alpha_sl = jnp.arctan2(3 * fy_max, cc)

    return jnp.where(
        jnp.abs(alpha) < alpha_sl,
        (
            -cc * jnp.tan(alpha)
            + (cc**2 / (3 * fy_max)) * jnp.abs(jnp.tan(alpha)) * jnp.tan(alpha)
            - (cc**3 / (27 * fy_max**2)) * jnp.tan(alpha) ** 3
        ),
        -fy_max * jnp.sign(alpha),
    )


class DynamicTrailerBicycleModel:
    def __init__(self, config: TrailerBicycleEnvConfig):
        self.config = config

    def initial_state(
        self,
        track: TrackModel,
        progress: float = 0.0,
        lateral_error: float = 0.0,
        heading_error: float = 0.0,
        speed: float = 6.0,
    ) -> VehicleState:
        x, y, yaw = track.spawn_pose(
            progress, lateral_error=lateral_error, heading_error=heading_error
        )

        return VehicleState(
            x=x,
            y=y,
            yaw_truck=yaw,
            yaw_trailer=yaw,
            vx=speed,
            vy=0.0,
            yaw_truck_rate=0.0,
            yaw_trailer_rate=0,
            steer=0.0,
            accel=0.0,
        )

    def step(self, state: VehicleState, action: jnp.ndarray, track: TrackModel) -> VehicleState:
        pass # math

class TrailerBicycleEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array_follow", "rgb_array_birds_eye"],
        "render_fps": 20,
    }

    def __init__(
        self,
        renderer: str | None = None,
        render_mode: str | None = None,
        render_width: int = 1280,
        render_height: int = 720,
    ) -> None:
        super().__init__()

        self.scenario = TrailerBicycleEnvConfig(
            "aach aach aach", TrackConfig(), VehicleConfig(), SimulationConfig()
        )
        self.track = TrackModel.from_config(self.scenario.track)
        self.dynamics = DynamicTrailerBicycleModel(self.scenario)

        self.renderer_kind = renderer
        self.render_mode = render_mode
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        if self.render_width <= 0 or self.render_height <= 0:
            raise ValueError("render_width and render_height must be positive integers.")
        self.renderer = None

        # self.runtime_track_id, self.runtime_car_id = self._resolve_runtime_ids()

        self._state: VehicleState | None = None

        obs_dim = 7

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _initial_state(
        self,
        initial_progress: float | None = None,
        initial_lateral_error: float | None = None,
        initial_heading_error: float | None = None,
        initial_speed: float | None = None,
    ) -> VehicleState:
        if any(
            value is not None
            for value in (
                initial_progress,
                initial_lateral_error,
                initial_heading_error,
                initial_speed,
            )
        ):
            progress = float(initial_progress if initial_progress is not None else 0.0) % 1.0
            lateral_error = float(
                initial_lateral_error if initial_lateral_error is not None else 0.0
            )
            heading_error = float(
                initial_heading_error if initial_heading_error is not None else 0.0
            )
            speed = float(initial_speed if initial_speed is not None else 8.0)
            return self.dynamics.initial_state(
                self.track,
                progress=progress,
                lateral_error=lateral_error,
                heading_error=heading_error,
                speed=speed,
            )

        return self.dynamics.initial_state(self.track, progress=0.0, speed=8.0)

    def _observation(self) -> np.ndarray:
        assert self._state is not None

        obs = np.concatenate(
            [
                np.array(
                    [
                        # self._state.progress,
                        # self._state.lateral_error,
                        # self._state.heading_error,
                        self._state.vx,
                        self._state.vy,
                        self._state.yaw_truck_rate,
                        self._state.yaw_trailer_rate,
                        self._state.yaw_truck,
                        self._state.yaw_trailer,
                        # self.track.sample(self._state.progress).curvature,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return obs

    def _render_state(self) -> dict[str, Any]:
        assert self._state is not None
        return {
            "x": self._state.x,
            "y": self._state.y,
            "yaw_truck": self._state.yaw_truck,
            "yaw_trailer": self._state.yaw_trailer,
            "steering_angle": self._state.steer * self.scenario.vehicle.max_steer_rad,
            "speed": self._state.vx,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        start_mode = options.get("start_mode", options.get("mode", "grid"))

        self._state = self._initial_state(
            initial_progress=options.get("initial_progress"),
            initial_lateral_error=options.get("initial_lateral_error"),
            initial_heading_error=options.get("initial_heading_error"),
            initial_speed=options.get("initial_speed"),
        )

        self._previous_feature_state = None
        self._step_count = 0
        self._lap_count = 0

        obs = self._observation()
        info = {
            "state": self._render_state(),
            "render_state": self._render_state(),
            "reset": {
                "start_mode": start_mode,
                "initial_progress": (
                    None
                    if options.get("initial_progress") is None
                    else float(options["initial_progress"])
                ),
                "initial_lateral_error": (
                    None
                    if options.get("initial_lateral_error") is None
                    else float(options["initial_lateral_error"])
                ),
                "initial_heading_error": (
                    None
                    if options.get("initial_heading_error") is None
                    else float(options["initial_heading_error"])
                ),
                "initial_speed": (
                    None
                    if options.get("initial_speed") is None
                    else float(options["initial_speed"])
                ),
            },
        }
        return obs, info

    def step(self, action):
        assert self._state is not None, "Call reset() before step()."
        action = np.asarray(action, dtype=float)
        previous_progress = self.track.project(self._state.x, self._state.y).progress

        self._state = self.dynamics.step(self._state, action, self.track)

        self._step_count += 1
        projection = self.track.project(self._state.x, self._state.y)
        if projection.progress < previous_progress - 0.5:
            self._lap_count += 1

        render_state = self._render_state()
        info = {
            "state": render_state,
            "render_state": render_state,
            "lap_count": self._lap_count,
        }
        terminated = self.track.out_of_bounds(projection.lateral_error)
        return self._observation(), 0, terminated, False, info

    def render(self):
        if self.render_mode is None or self.renderer_kind != "pybullet":
            return None
        if self.renderer is None:
            self.renderer = PyBulletMirrorRenderer(
                self.scenario,
                self.track,
                self.render_mode,
                width=self.render_width,
                height=self.render_height,
            )
        return self.renderer.render(self._render_state())

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
