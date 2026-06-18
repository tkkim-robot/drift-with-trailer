from uncertain_racecar_gym.jax_env import build_nominal_jax_params, NominalJaxRacecarEnv
from uncertain_racecar_gym.env import VehicleState
import gymnasium as gym
from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.controllers.mpc.smppi_jax import SMPPI_Jax
from src.dynamics.vehicle.bicycle_fiala import gen_util_funs
import time
import cv2
from gymnasium.wrappers import RecordVideo
from src.simulation.bicycle_env import BicycleEnv

import jax.numpy as jnp


def run_mpc(mpc, env, config):
    speeds, slip_angles_f, slip_angles_r, yaw_rates = [], [], [], []
    env.reset()

    observation, reward, terminated, truncated, info = env.step(jnp.zeros(3))
    i = 0
    try:
        while True:
            start = time.perf_counter()

            state: VehicleState = env.unwrapped._state

            mpc_state = jnp.array(
                [
                    state.x,
                    state.y,
                    state.yaw,
                    state.vx,
                    state.vy,
                    state.yaw_rate,
                    env.unwrapped.track.find_mu(state.x, state.y),
                    env.unwrapped.track._arc_samples[env.unwrapped._last_index],
                ]
            )

            u = mpc.run_mpc(mpc_state)
            u.block_until_ready()


            # Benchmarking
            speeds.append(jnp.hypot(state.vx, state.vy))
            yaw_rates.append(state.yaw_rate)

            vx_safe = jnp.maximum(jnp.abs(state.vx), 0.5)
            steer_angle = state.steer * env.unwrapped.scenario.vehicle.max_steer_rad
            alpha_f = steer_angle - jnp.arctan2(
                state.vy + env.unwrapped.scenario.vehicle.lf * state.yaw_rate, vx_safe
            )
            alpha_r = -jnp.arctan2(
                state.vy - env.unwrapped.scenario.vehicle.lr * state.yaw_rate, vx_safe
            )

            slip_angles_f.append(alpha_f)
            slip_angles_r.append(alpha_r)

            action = jnp.array([u[0], u[1]])
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated:  # or truncated:
                break

            i += 1
            frame = env.render()
            cv2.imshow("sim", frame[..., ::-1])
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass

    cutoff = 100
    print(
        f"Config: {config} \t| "
        f"Avg speed: {jnp.mean(jnp.array(speeds[cutoff:])) * 3.6:<7.3f} | "
        f"Avg alpha_f: {jnp.mean(jnp.array(slip_angles_f[cutoff:])):<7.3f} | "
        f"Avg alpha_r: {jnp.mean(jnp.array(slip_angles_r[cutoff:])):<7.3f} | "
        f"Avg yaw_rate: {jnp.mean(jnp.array(yaw_rates[cutoff:])):<7.3f} | "
        f"End step: {i:<5d}"
    )


if __name__ == "__main__":
    scenario = "ks_barcelona_layout_gp_dallara_f317_rl_long.yaml"
    # scenario = "sample_oval.yaml"

    env = BicycleEnv(
        renderer="pybullet",
        render_mode="rgb_array_birds_eye",
        render_width=300,
        render_height=200,
    )

    trials = [
        ("MPPI For.; no v bound", "SMPPI For.;no v bound", False, None),
        ("MPPI Rev.; no v bound", "SMPPI Rev.;no v bound", True, None),
        ("MPPI For.;  v_t = 90 ", "SMPPI For.; v_t = 90 ", False, 90),
        ("MPPI For.;  v_t = 120", "SMPPI For.; v_t = 120", False, 120),
        ("MPPI For.;  v_t = 250", "SMPPI For.; v_t = 250", False, 250),
        ("MPPI Rev.;  v_t = 100", "SMPPI Rev.; v_t = 100", False, -100),
        ("MPPI Rev.;  v_t = 250", "SMPPI Rev.; v_t = 250", False, -250),
    ]

    for config_mppi, config_smppi, reverse, v_t in trials:
        dynamics, cost, bound, bound_der = gen_util_funs(env.unwrapped.scenario, reverse=reverse, v_target=(None if v_t is None else v_t / 3.6))

        mppi = MPPI_Jax(
            6,
            2,
            dynamics,
            None,
            cost,
            bound,
            jnp.diag(jnp.array([0.0625, 1])),
            inverse_temp=1e-1,
            K=1000,
            gamma=5,
            step=0.05,
            T=100,
        )
        smppi = SMPPI_Jax(
            6,
            2,
            dynamics,
            None,
            cost,
            bound,
            bound_der,
            jnp.diag(jnp.array([1 / 256, 1])), # 0.25, 0.75
            jnp.diag(jnp.array([1e-1, 1e-2])),
            inverse_temp=1,
            K=1000,
            gamma=1,
            step=0.05,
            T=100,
        )

        run_mpc(mppi, env, config_mppi)
        run_mpc(smppi, env, config_smppi)