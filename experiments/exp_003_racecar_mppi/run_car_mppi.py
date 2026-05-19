from uncertain_racecar_gym.jax_env import build_nominal_jax_params
import gymnasium as gym
from src.controllers.mpc.mppi_jax import MPPI_Jax
from experiments.exp_003_racecar_mppi.dynamics import gen_util_funs

import numpy as np
import time

import jax.numpy as jnp
import jax

def run_mpc():
    env = gym.make(
        "UncertainRacecar-v0",
        scenario="package://scenarios/ks_barcelona_layout_gp_dallara_f317_rl_long.yaml",
        uncertainty=None,
        renderer="pybullet",
        render_mode="rgb_array_follow",
    )
    env.reset()

    params = build_nominal_jax_params("package://scenarios/ks_barcelona_layout_gp_dallara_f317_rl_long.yaml")
    dynamics, cost, bound = gen_util_funs(params)

    mpc = MPPI_Jax(4, 1, dynamics, lambda _, __: 0, cost, bound)
    
    observation, reward, terminated, truncated, info = env.step(0)

    i = 0

    while True:
        start = time.perf_counter()
        u = mpc.run_mpc(observation)
        action = np.array(u[0])
        print(i, time.perf_counter() - start, action)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated: 
            break
        i += 1


    env.close()

if __name__ == "__main__":
    run_mpc()