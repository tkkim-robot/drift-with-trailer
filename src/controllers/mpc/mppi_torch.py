import torch
from scipy.signal import savgol_filter
from scipy.stats import norm
import numpy as np



class MPPI_Torch:
    """
    Torch MPPI
    """

    def __init__(
        self,
        x_d,
        u_d,
        dynamics_func,
        term_cost_func,
        cost_func,
        inverse_temp=12.5,
        alpha=0.99,
        gamma=0.1,
        K=1000,
        step=0.02 * 2,
        n=20,
    ):
        """ 
        """
        self.last_trajectory = None
        self.dynamics = dynamics_func
        self.term_cost = term_cost_func
        self.cost = cost_func
        self.alpha = alpha
        self.gamma = gamma
        self.inverse_temp = inverse_temp
        self.K = K

        self.x_d = x_d
        self.u_d = u_d
        self.T = n

        self.step = step
        self.cv = torch.eye(u_d) * 5 

        self.inv_cv = torch.inverse(self.cv)

    def _forward_sim(self, x: torch.Tensor, u: torch.Tensor, noise: torch.Tensor):
        """
        Uses Euler's method to integrate the dynamics

        Args:
            x (ca.SX): States
            u (ca.SX): Controls

        Returns:
            ca.SX: Next state
        """
        v = u + noise

        S = torch.zeros(self.K)

        for i in range(self.T):
            x += self.dynamics(x, v[i, :]) * self.step
            S += (
                self.cost(x, v[i, :], i)
                - self.gamma # why subtract??
                * torch.transpose(u[i, :].unsqueeze(0), 0, 1)
                @ self.inv_cv
                @ noise[i, :]
            )  # check last indexing, fix wrong matmul

        S += self.term_cost(x, u[-1, :])
        return S

    def _weights(self, costs: torch.Tensor) -> torch.Tensor:
        weights = torch.exp(-(costs - costs.min()) / self.inverse_temp)
        return weights / weights.sum()

    # def setup_mpc(self):
    #     """
    #     Formulates problem, sets constraints

    #     Returns:
    #         ca.SX: Controls (u)
    #     """
    #     x = self.x0
    #     u = self.opti.variable(self.n, self.u_d)
    #     J = 0

    #     for i in range(self.n):
    #         x = self._euler_step(x, u[i])

    #         J += self.cost(x, u[i])
    #         self.constraint(self.opti, x, u[i])

    #     if self.term_cost:
    #         J += self.term_cost(x, u[0], u[-1])

    #     self.opti.minimize(J)

    #     if self.term_constraint:
    #         self.term_constraint(self.opti, x, u[0], u[-1])

    #     return u

    def run_mpc(self, x, verbose=True, warm_start=True):
        """
        Runs a single MPC solve.
        Args:
            x (np.ndarray): Measured stat
            verbose (bool, optional): _description_. Defaults to True.
            warm_start (bool, optional): _description_. Defaults to True.

        Returns:
            ca.SX: u
        """

        u = torch.zeros(self.u_d)

        if warm_start and self.last_trajectory is not None:
            u[1:] = self.last_trajectory[1:]

        x_batch = x.unsqueeze(0).repeat(self.K, 1)
        u_batch = u.unsqueeze(0).repeat(self.K, 1)

        noise = torch.from_numpy(
            np.random.multivariate_normal(
                np.zeros(self.u_d), self.cv.numpy(), size=(self.T, self.K)
            )
        )

        costs = self._forward_sim(x_batch, u_batch, noise)

        weights = self._weights(costs)

        weighted_noise = weights * noise

        filtered = savgol_filter(
            weighted_noise, window_length=5, polyorder=3, axis=0
        )  # no idea if correct

        u += filtered

        self.last_trajectory = u

        return u[0]
