import cv2
import numpy as np
import time
import jax.numpy as jnp
import jax
from gymnasium.wrappers import RecordVideo

from src.simulation.cartpole_env import CartPoleEnv
from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.learning.models.cartpole_nn import CartpoleModel

from experiments.exp_006_learned_cartpole_dynamics.cartpole_utils import (
    term_cost,
    cost,
    bound_control,
    FORCE,
)
from experiments.exp_006_learned_cartpole_dynamics.cartpole_nn_dynamics import LearnedDynamics

env = CartPoleEnv(render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="gym_videos",
    episode_trigger=lambda x: True,
    disable_logger=True,
    name_prefix="cartpole_nn",
)

env.reset()

BATCH_SIZE = 16
EPOCHS = 30
LR = 0.005
N = 64  # preferably a multiple of BATCH_SIZE


dynamics = LearnedDynamics(
    CartpoleModel(5, 4),
    BATCH_SIZE,
    state_mean=jnp.zeros(5),
    state_std=jnp.array([3, 3, 1, 1, 3]),
    dynamics_mean=jnp.zeros(4),
    dynamics_std=jnp.array([0.05, 0.1, 0.05, 0.2]),
    optimizer_params=dict(learning_rate=LR),
)

device = "cpu"

mpc = MPPI_Jax(4, 1, dynamics, term_cost, cost, bound_control, jnp.eye(1) * 3)

observation, reward, terminated, truncated, info = env.step(0)

i = 0

try:
    while True:
        start = time.perf_counter()
        u = mpc.run_mpc(observation)
        action = np.clip(float(np.array(u[0])), -FORCE, FORCE)
        print(i, time.perf_counter() - start, action)

        next_observation, reward, terminated, truncated, info = env.step(action)
        dynamics.data.add(
            jnp.array([*observation, action]), jnp.array(next_observation - observation) / 0.02
        )

        observation = next_observation
        i += 1

        frame = env.render()
        cv2.imshow("sim", frame[..., ::-1])
        cv2.waitKey(1)

        if terminated:
            print("Died, resetting")
            observation, _ = env.reset()
            continue

        if i % N == 0:
            dynamics.train(EPOCHS)
            jax.clear_caches() # in case jax is not using the updated model?


finally:
    env.close()
