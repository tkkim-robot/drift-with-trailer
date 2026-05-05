import torch
from scipy.signal import savgol_filter
from scipy.stats import norm
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal



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
        bound_control_func,
        inverse_temp=1,
        alpha=0.01,
        gamma=0.1,
        K=20000,
        step=0.02,
        T=100,
        device="mps"
    ):
        """ 
        """
        self.last_trajectory = None
        self.dynamics = dynamics_func
        self.term_cost = term_cost_func
        self.cost = cost_func
        self.bound_control = bound_control_func
        self.alpha = alpha
        self.inverse_temp = inverse_temp
        self.gamma = gamma
        self.K = K
        self.device = torch.device(device)

        self.x_d = x_d
        self.u_d = u_d
        self.T = T

        self.step = step
        self.cv = torch.eye(u_d, device=device) * 10

        self.inv_cv = torch.inverse(self.cv)

        self.dist = MultivariateNormal(torch.zeros(self.u_d, device=device), self.cv)


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
        prev = round(self.K * (1 - self.alpha))
        v[:, prev:] = noise[:, prev:]
        v = self.bound_control(v)
        noise.copy_(v - u)

        S = torch.zeros(self.K, device=self.device)

        for i in range(self.T):
            x += self.dynamics(x, v[i, :]) * self.step
            
            # print(u.device, self.inv_cv.device, noise.device)
            S += (
                self.cost(x, v[i, :], i)
                - self.gamma
                * (u[i, :].unsqueeze(1) @ self.inv_cv @ noise[i, :].unsqueeze(2)).squeeze(-1).squeeze(-1)
            )

        if self.term_cost:
            S += self.term_cost(x, u[-1, :])
        return S

    def _weights(self, costs: torch.Tensor) -> torch.Tensor:
        weights = torch.exp(-(costs - costs.min()) / self.inverse_temp)
        return weights / weights.sum()
    

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

        u = torch.zeros(self.T, self.u_d, device=self.device)

        if warm_start and self.last_trajectory is not None:
            u[:-1] = self.last_trajectory[1:]

        x = torch.from_numpy(x).to(self.device)

        x_batch = x.unsqueeze(0).repeat(self.K, 1)
        u_batch = u.unsqueeze(0).repeat(self.K, 1, 1).permute(1, 0, 2) # this is terrible

        noise = self.dist.sample(u_batch.shape[:-1])

        costs = self._forward_sim(x_batch, u_batch, noise)

        weights = self._weights(costs)
        weighted_noise = torch.sum(weights.view(1, -1, 1) * noise, dim=1)
        u += weighted_noise

        self.last_trajectory = u

        return u[0].cpu()
