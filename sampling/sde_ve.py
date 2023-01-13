from typing import Union

import torch

from sampling.samplers_abc import DeterministicEulerSampler
from utils import append_zero


class VESampler(DeterministicEulerSampler):
    def __init__(
        self,
        N: int,
        device: Union[str, torch.device],
        sigma_min=0.02,
        sigma_max=100,
    ) -> None:
        super().__init__(N, device)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _set_timesteps(self) -> None:
        step_i = torch.linspace(0, 1, self.N, device=self.device)
        sigma_max_sq = self.sigma_max**2
        sigma_min_sq = self.sigma_min**2

        timesteps = sigma_max_sq * (sigma_min_sq / sigma_max_sq) ** step_i

        # t_N = 0 added for consistency with paper, but never accessed in the sampling code
        self.timesteps = append_zero(timesteps)

    def _get_sigma(self, i):
        sigma_i = torch.sqrt(self.timesteps[i])
        return sigma_i

    def _get_scale(self, i):
        return 1.0

    def _get_sigma_derivative(self, i):
        return 0.5 / self._get_sigma(i)

    def _get_scale_derivative(self, i):
        return 0.0
