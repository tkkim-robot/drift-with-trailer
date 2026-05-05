from src.simulation.cartpole_env import CartPoleEnv
from src.controllers.mpc.mppi_torch import MPPI_Torch

import casadi as ca
import numpy as np
import torch

# Constants
POLE_LEN = 0.5
FORCE = 10

g = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
TOTAL_MASS = POLE_MASS + CART_MASS
POLEMASS_LENGTH = POLE_MASS * POLE_LEN

def gen_dynamics(x, u):
    x_c, d_x_c, theta, theta_dot = x

    # x, x_dot, theta, theta_dot = self.state
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    temp = (
        u + POLE_LEN * (theta_dot * theta_dot) * sintheta
    ) / TOTAL_MASS

    thetaacc = (g * sintheta - costheta * temp) / (
        POLE_LEN
        * (4.0 / 3.0 - POLE_MASS * (costheta * costheta) / TOTAL_MASS)
    )

    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    dx = d_x_c, xacc, theta_dot, thetaacc

    return dx

def constraints(x, u):
    x_threshold = 2.4

    return -x_threshold <= x[0] <= x_threshold and -FORCE <= u <= FORCE

def cost(x, u, t):
    return x[2] ** 2 + x[0] ** 2 + 0.9**t * (10_000 if constraints(x, u) else 0)

def run_mpc():
    env = CartPoleEnv(render_mode="human")

    env.reset()

    mpc = MPPI_Torch(4, 1, gen_dynamics, None, cost, constraints)
    
    observation, reward, terminated, truncated, info = env.step(0)

    while True:
        u = mpc.run_mpc(observation)
        observation, reward, terminated, truncated, info = env.step(np.clip(u, -FORCE, FORCE))

        if terminated: 
            break


    env.close()

if __name__ == "__main__":
    run_mpc()