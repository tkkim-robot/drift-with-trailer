from scipy.signal import savgol_filter
import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
import numpy as np
import torch
import functools

# @jax.jit
@functools.partial(
    jax.jit, 
    static_argnames=["cost", "term_cost", "bound_control", "dynamics", "K"]
)
@functools.partial(
    jax.vmap, 
    in_axes=(0, 0, 0, None, None, None, None, None, None, None, None)
)
def rollout(
    x: ArrayLike,
    u: ArrayLike,
    noise: ArrayLike,
    K,
    gamma,
    inv_cv,
    cost,
    term_cost,
    bound_control,
    dynamics,
    step,
) -> float:
    """
    Uses Euler's method to integrate the dynamics

    Args:
        x (ArrayLike): State (T, x_d)
        u (ArrayLike): Control (T, u_d)
        noise (ArrayLike): Noise (T, u_d)

    Returns:
        float: Cost of rollout
    """

    v = u + noise
    v = bound_control(v)
    bounded_noise = v - u

    S = jnp.zeros(K)

    def step_dynamics(carry, control):
        x, S, i = carry
        u, v, bounded_noise = control

        new_x = x + dynamics(x, v) * step
        new_S = S + cost(x, v, i) - gamma * jnp.einsum("n,nm,m->", u, inv_cv, bounded_noise)
        new_i = i + 1

        new_carry = new_x, new_S, new_i

        return new_carry, (new_x, new_S)

    (x, S, _), _ = jax.lax.scan(
        step_dynamics, (x, 0, 0), (u, v, bounded_noise)
    )

    # for i in range(T):
    #     x += dynamics(x, v[i, :]) * step

    #     # print(u.device, self.inv_cv.device, noise.device)
    #     S += cost(x, v[i, :], i) - gamma * (
    #         u[i, :].unsqueeze(1) @ inv_cv @ noise[i, :].unsqueeze(2)
    #     ).squeeze(-1).squeeze(-1)

    if term_cost:
        S += term_cost(x, u[-1])
    return S


class MPPI_Jax:
    """
    JAX MPPI
    """

    def __init__(
        self,
        x_d: int,
        u_d: int,
        dynamics_func,
        term_cost_func,
        cost_func,
        bound_control_func,
        inverse_temp=1,
        alpha=0.01,
        gamma=0.01,
        K=5000,
        step=0.02,
        T=70,
        device="mps",
    ):
        """
        Args:
            x_d (int): State dimension
            u_d (int): Control dimension
            dynamics_func (Callable): Dynamics function
            term_cost_func (Callable): Terminal cost function
            cost_func (Callable): Cost function
            bound_control_func (Callable): Function that bounds controls
            inverse_temp (int, optional): Actually the temperature. Defaults to 1.
            alpha (float, optional): Proportion of samples set to just noise. Defaults to 0.01.
            gamma (float, optional): Cost weight for pertubations. Defaults to 0.1.
            K (int, optional): Samples. Defaults to 5000.
            step (float, optional): Time step. Defaults to 0.02.
            T (int, optional): Time horizon in steps. Defaults to 50.
        """
        self.last_trajectory = None
        self.u_history = jnp.zeros((T, u_d))
        self.dynamics = dynamics_func
        self.term_cost = term_cost_func
        self.cost = cost_func
        self.bound_control = bound_control_func
        self.alpha = alpha
        self.inverse_temp = inverse_temp
        self.gamma = gamma
        self.K = K
        self.device = device

        self.x_d = x_d
        self.u_d = u_d
        self.T = T

        self.step = step
        self.cv = jnp.eye(u_d) * 20

        self.inv_cv = jnp.linalg.inv(self.cv)

        self.key = jax.random.key(0)

    def _forward_sim(self, x: ArrayLike, u: ArrayLike, noise: ArrayLike) -> jax.Array:
        """
        Uses Euler's method to integrate the dynamics

        Args:
            x (ArrayLike): State (T, K, x_d)
            u (ArrayLike): Control (T, K, u_d)
            noise (ArrayLike): Noise (T, K, u_d)

        Returns:
            jax.Array: Cost per sample (K)
        """

        v = u + noise
        prev = round(self.K * (1 - self.alpha))

        # v[:, prev:] = noise[:, prev:]
        v = v.at[:, prev:].set(noise[:, prev:])

        v = self.bound_control(v)
        noise = v - u

        S = rollout(
            x,
            u,
            noise,
            self.K,
            self.gamma,
            self.inv_cv,
            self.cost,
            self.term_cost,
            self.bound_control,
            self.dynamics,
            self.step,
        )

        return S

    def _weights(self, costs: ArrayLike) -> jax.Array:
        """
        Computes weights

        Args:
            costs (torch.Tensor): Costs (K)

        Returns:
            torch.Tensor: Weights (K)
        """
        weights = jnp.exp(-(costs - costs.min()) / self.inverse_temp)
        return weights / weights.sum()

    def run_mpc(self, x: ArrayLike, verbose=True) -> torch.Tensor:
        """
        Runs a single MPC solve.

        Args:
            x (torch.Tensor): State (x_d)
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            torch.Tensor: Control output
        """

        if self.last_trajectory is None:
            u = jnp.zeros((self.T, self.u_d))
        else:
            u = jnp.roll(self.last_trajectory, -1, axis=1)
            u = u.at[-1].set(0)

        x = jnp.asarray(x)

        # x_batch = x.unsqueeze(0).repeat(self.K, 1)
        # u_batch = u.unsqueeze(0).repeat(self.K, 1, 1)  # .permute(1, 0, 2) # this is terrible
        x_batch = jnp.repeat(jnp.expand_dims(x, 0), self.K, axis=0)
        u_batch = jnp.repeat(jnp.expand_dims(u, 0), self.K, axis=0)

        noise = jax.random.normal(self.key, u_batch.shape) * jnp.sqrt(self.cv)

        costs = self._forward_sim(x_batch, u_batch, noise)

        weights = self._weights(costs)

        weighted_noise = jnp.sum(weights.reshape(-1, 1, 1) * noise, axis=0)
        u = u + weighted_noise

        u_padded = jnp.concatenate([self.u_history, u])
        u_smoothed = jnp.array(
            savgol_filter(np.array(u_padded), 5, 3, axis=0)[-self.T :]
        )

        self.u_history = jnp.roll(self.u_history, -1, axis=0)
        self.u_history = self.u_history.at[-1].set(u_smoothed[0])
        self.last_trajectory = u

        return u_smoothed[0]
