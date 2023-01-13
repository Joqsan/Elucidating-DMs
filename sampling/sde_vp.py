from typing import Union

import torch

from sampling.samplers_abc import DeterministicEulerSampler
from utils import append_zero


class VPSampler(DeterministicEulerSampler):
    def __init__(
        self,
        N: int,
        device: Union[str, torch.device],
        beta_d=19.9,
        beta_min=0.1,
        eps_s=1e-3,
        eps_t=1e-5,
        M=100,
    ) -> None:
        super().__init__(N, device)
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.eps_s = eps_s
        self.eps_t = eps_t
        self.M = M

    def _set_timesteps(self) -> None:
        step_i = torch.linspace(0, 1, self.N, device=self.device)

        timesteps = 1 + step_i * (self.eps_s - 1)

        # t_N = 0 added for consistency with paper, but never accessed in the sampling code
        self.timesteps = append_zero(timesteps)

    def _get_sigma(self, i):
        exponent_i = self._get_exponent(i)
        sigma_i = torch.sqrt(torch.exp(exponent_i) - 1)
        return sigma_i

    def _get_scale(self, i):
        exponent_i = self._get_exponent(i)
        inv_scale_i = torch.sqrt(torch.exp(exponent_i))
        return 1 / inv_scale_i

    def _get_sigma_derivative(self, i):
        exponent_i = self._get_exponent(i)
        d_exponent_i = self._get_exponent_derivative(i)
        dsigma_i = (
            d_exponent_i * torch.exp(exponent_i) / (2 * torch.sqrt(exponent_i - 1))
        )

        return dsigma_i

    def _get_scale_derivative(self, i):
        exponent_i = self._get_exponent(i)
        d_exponent_i = self._get_exponent_derivative(i)
        d_scale_i = -1 * d_exponent_i / (2 * torch.sqrt(exponent_i))

    def _get_exponent(self, i):
        t_i = self.timesteps[i]
        return self.beta_d * t_i**2 / 2 + self.beta_min * t_i

    def _get_exponent_derivative(self, i):
        t_i = self.timesteps[i]
        return self.beta_d * t_i + self.beta_min
