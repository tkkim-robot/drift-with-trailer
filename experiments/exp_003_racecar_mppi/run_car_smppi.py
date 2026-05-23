from uncertain_racecar_gym.jax_env import build_nominal_jax_params
from uncertain_racecar_gym.env import VehicleState
import gymnasium as gym
from src.controllers.mpc.smppi_jax import SMPPI_Jax
from experiments.exp_003_racecar_mppi.dynamics import gen_util_funs

import time
import cv2
from gymnasium.wrappers import RecordVideo


import jax.numpy as jnp


def run_mpc(scenario):

    env = RecordVideo(
        gym.make(
            "UncertainRacecar-v0",
            scenario=f"package://scenarios/{scenario}",
            uncertainty=None,
            renderer="pybullet",
            render_mode="rgb_array_follow",
        ),
        video_folder="gym_videos",
        episode_trigger=lambda x: True,
    )
    env.reset()

    params = build_nominal_jax_params(
        scenario=f"package://scenarios/{scenario}",
    )
    dynamics, cost, bound = gen_util_funs(params[0])

    mpc = SMPPI_Jax(6, 3, dynamics, None, cost, bound, jnp.diag(jnp.array([1, 0.5, 0.5])))

    observation, reward, terminated, truncated, info = env.step(jnp.zeros(3))

    i = 0
    try:
        while True:
            start = time.perf_counter()

            state: VehicleState = env.unwrapped._state

            mpc_state = jnp.array([state.x, state.y, state.yaw, state.vx, state.vy, state.yaw_rate])

            u = mpc.run_mpc(mpc_state)

            print(i, time.perf_counter() - start, u, round(state.progress, 3), round(state.vx, 3), round(state.vy, 3))

            observation, reward, terminated, truncated, info = env.step(u)

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
    scenario = "sample_oval.yaml"

    run_mpc(scenario)
