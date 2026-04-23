from src.simulation.cartpole_env import CartPoleEnv
from src.controllers.mpc.ipopt_cartpole import MPC

import casadi as ca
import numpy as np

# Constants
POLE_LEN = 0.5
FORCE = 10

g = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
TOTAL_MASS = POLE_MASS + CART_MASS
POLEMASS_LENGTH = POLE_MASS * POLE_LEN


def gen_dynamics():
    x = ca.SX.sym("x", 4)
    u = ca.SX.sym("u")

    x_c, d_x_c, theta, theta_dot = ca.vertsplit(x)

    # x, x_dot, theta, theta_dot = self.state
    costheta = ca.cos(theta)
    sintheta = ca.sin(theta)


    temp = (
        u + POLE_LEN * (theta_dot * theta_dot) * sintheta
    ) / TOTAL_MASS
    thetaacc = (g * sintheta - costheta * temp) / (
        POLE_LEN
        * (4.0 / 3.0 - POLE_MASS * (costheta * costheta) / TOTAL_MASS)
    )
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    dx = ca.vertcat(d_x_c, xacc, theta_dot, thetaacc)

    dynam_f = ca.Function("dynamics", [x, u], [dx])
    return dynam_f

def constraints(opti, x, u):
    x_threshold = 2.4

    opti.subject_to(opti.bounded(-x_threshold, x[0], x_threshold))
    opti.subject_to(opti.bounded(-FORCE, u, FORCE))

def cost(x, u):
    return x[2] ** 2 + x[0] ** 2

ipopt_settings = {
    "record_time": True,
    "ipopt.print_frequency_iter": 25,
    "ipopt.print_level": 0,
    "print_time": 0,
    "ipopt.sb": "no",
    "ipopt.max_iter": 5000,
    "detect_simple_bounds": True,
    "ipopt.linear_solver": "ma97",
    "ipopt.mu_strategy": "adaptive",
    "ipopt.nlp_scaling_method": "gradient-based",
    "ipopt.bound_relax_factor": 1e-4,
    # "ipopt.hessian_approximation": "exact",
    "ipopt.tol": 1e-4,
    "ipopt.hessian_approximation": "limited-memory",
    "ipopt.limited_memory_max_history": 10,
    "ipopt.limited_memory_update_type": "bfgs",
    "ipopt.derivative_test": "none",
}

def run_mpc():
    env = CartPoleEnv(render_mode="human")

    env.reset()

    mpc = MPC(4, 1, gen_dynamics, None, constraints, None, cost, ipopt_settings)
    observation, reward, terminated, truncated, info = env.step(0)

    while True:
        u = mpc.run_mpc(observation)
        observation, reward, terminated, truncated, info = env.step(np.clip(u, -FORCE, FORCE))

        if terminated: 
            break


    env.close()

if __name__ == "__main__":
    run_mpc()