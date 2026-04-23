"""Small cartpole smoke simulation for checking the uv environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartPoleParams:
    gravity: float = 9.81
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    half_pole_length: float = 0.5
    force_limit: float = 10.0
    dt: float = 0.02


def cartpole_dynamics(
    state: np.ndarray, force: float, params: CartPoleParams
) -> np.ndarray:
    """Return time derivative for [x, x_dot, theta, theta_dot]."""
    x, x_dot, theta, theta_dot = state
    del x

    force = float(np.clip(force, -params.force_limit, params.force_limit))
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    total_mass = params.mass_cart + params.mass_pole
    polemass_length = params.mass_pole * params.half_pole_length

    temp = (force + polemass_length * theta_dot**2 * sin_theta) / total_mass
    theta_acc = (params.gravity * sin_theta - cos_theta * temp) / (
        params.half_pole_length
        * (4.0 / 3.0 - params.mass_pole * cos_theta**2 / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass

    return np.array([x_dot, x_acc, theta_dot, theta_acc], dtype=float)


def rk4_step(state: np.ndarray, force: float, params: CartPoleParams) -> np.ndarray:
    """Advance cartpole dynamics by one RK4 step."""
    dt = params.dt
    k1 = cartpole_dynamics(state, force, params)
    k2 = cartpole_dynamics(state + 0.5 * dt * k1, force, params)
    k3 = cartpole_dynamics(state + 0.5 * dt * k2, force, params)
    k4 = cartpole_dynamics(state + dt * k3, force, params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def demo_force(t: float) -> float:
    """A tiny open-loop force just to make the smoke simulation move."""
    return 2.0 * np.sin(2.0 * np.pi * 0.5 * t)


def main() -> None:
    params = CartPoleParams()
    state = np.array([0.0, 0.0, np.pi - 0.15, 0.0], dtype=float)
    steps = 100

    for step in range(steps):
        force = demo_force(step * params.dt)
        state = rk4_step(state, force, params)

    if not np.all(np.isfinite(state)):
        raise RuntimeError("Cartpole smoke simulation produced a non-finite state.")

    print("Cartpole smoke simulation completed.")
    print(f"final_state = {np.array2string(state, precision=4)}")


if __name__ == "__main__":
    main()
