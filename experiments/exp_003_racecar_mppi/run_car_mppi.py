from uncertain_racecar_gym.jax_env import build_nominal_jax_params, NominalJaxRacecarEnv
from uncertain_racecar_gym.env import VehicleState
import gymnasium as gym
from src.controllers.mpc.mppi_jax import MPPI_Jax
from experiments.exp_003_racecar_mppi.dynamics import gen_util_funs

import time
import cv2
from gymnasium.wrappers import RecordVideo


import jax.numpy as jnp


def run_mpc():
    scenario = "sample_oval.yaml" # sample_oval.yaml ks_barcelona_layout_gp_dallara_f317_rl_long

    env = RecordVideo(
        gym.make(
            "UncertainRacecar-v0",
            scenario=f"package://scenarios/{scenario}",
            uncertainty=None,
            renderer="pybullet",
            render_mode="rgb_array_follow",
            render_width=300,
            render_height=200,
        ),
        video_folder="gym_videos",
        episode_trigger=lambda x: True,
    )
    env.reset()

    params = build_nominal_jax_params(
        scenario=f"package://scenarios/{scenario}",
    )
    dynamics, cost, bound = gen_util_funs(params[0])

    mpc = MPPI_Jax(6, 3, dynamics, None, cost, bound, jnp.diag(jnp.array([0.25, 0.5, 0.5])), inverse_temp=0.5, K=1000, gamma=0.1, step = 0.05, T=30)

    observation, reward, terminated, truncated, info = env.step(jnp.zeros(3))

    i = 0
    try:
        while True:
            start = time.perf_counter()

            state: VehicleState = env.unwrapped._state

            mpc_state = jnp.array([state.x, state.y, state.yaw, state.vx, state.vy, state.yaw_rate])

            u = mpc.run_mpc(mpc_state)
            u.block_until_ready() 

            print(i, time.perf_counter() - start, u, round(state.progress, 3), round(state.vx, 3), round(state.vy, 3))
            a = time.perf_counter()
            observation, reward, terminated, truncated, info = env.step(u)
            print(time.perf_counter() - a)
            if terminated:
                break

            i += 1
            frame = env.render()
            cv2.imshow("sim", frame[..., ::-1])
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == "__main__":
    run_mpc()
