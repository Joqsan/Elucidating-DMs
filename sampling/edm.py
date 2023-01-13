from typing import Union

import torch

from sampling.samplers_abc import StochasticModifiedEulerSampler
from utils import append_zero


class EDMSampler(StochasticModifiedEulerSampler):
    def __init__(
        self,
        N: int,
        device: Union[str, torch.device],
        S_churn: float = 0,
        S_tmin: float = 0,
        S_tmax: float = float("inf"),
        S_noise: float = 1,
        sigma_min=0.002,
        sigma_max=80,
        rho=7.0,
    ) -> None:
        super().__init__(N, device, S_churn, S_tmin, S_tmax, S_noise)

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def _set_timesteps(self):
        # timesteps -- not discrete
        step_i = torch.linspace(0, 1, self.N, device=self.device)
        inv_rho = 1 / self.rho
        timesteps = (
            self.sigma_max**inv_rho
            + step_i * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        # t_N = 0 added for consistency with paper, but never accessed in the sampling code
        self.timesteps = append_zero(timesteps)

    def _get_sigma(self, i):
        sigma_i = self.timesteps[i]
        return sigma_i

    def _get_scale(self, i):
        return 1.0

    def _get_sigma_derivative(self, i):
        return 1.0

    def _get_scale_derivative(self, i):
        return 0.0
