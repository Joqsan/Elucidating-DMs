from abc import ABC, abstractmethod
from typing import Union

import torch


class DeterministicHeunSampler(ABC):
    def __init__(
        self,
        N: int,
        device: Union[str, torch.device],
    ) -> None:
        super().__init__()
        self.N = N
        self.device = device

        # [0, 0, 1, 1, 2, 2, ..., N-2, N-1]
        self.i_steps = torch.arange(0, N, device=self.device)
        self.i_steps = torch.cat(
            [self.i_steps[:-1].repeat_interleave(2), self.i_steps[-1:]]
        )

        self._set_timesteps()

        self.first_order_d_i = None

    @abstractmethod
    def _set_timesteps(self) -> None:
        """Compute self.timesteps such that t_i = self.timesteps[i]"""
        self.timesteps: torch.Tensor
        return

    @abstractmethod
    def _get_sigma(self, i) -> Union[float, torch.Tensor]:
        """Compute and return sigma_i = sigma(t_i)"""
        return

    @abstractmethod
    def _get_scale(self, i) -> Union[float, torch.Tensor]:
        """Compute and return scale_i = scale(t_i). In the paper scale(t_i) = s(t_i)"""
        return

    @abstractmethod
    def _get_sigma_derivative(self, i) -> Union[float, torch.Tensor]:
        """Compute and return (dsigma/dt)(t_i)"""
        return

    @abstractmethod
    def _get_scale_derivative(self, i) -> Union[float, torch.Tensor]:
        """Compute and return (dscale/dt)(t_i)"""
        return

    @property
    def in_first_order_step(self):
        return self.first_order_d_i is None

    @torch.no_grad()
    def step(
        self,
        i: Union[int, torch.Tensor],
        denoised_output_i: torch.Tensor,
        x_i: torch.Tensor,
    ):
        t_i = self.timesteps[i]
        t_i_next = self.timesteps[i + 1]
        sigma_i = self._get_sigma(i)
        scale_i = self._get_scale(i)
        d_sigma_i = self._get_sigma_derivative(i)
        d_scale_i = self._get_scale_derivative(i)

        ratio_sigma_i = (
            d_sigma_i / sigma_i
        )  # safe: sigma_i is never 0 for the given i's
        ratio_scale_i = d_scale_i / scale_i

        d_i = (
            ratio_sigma_i + ratio_scale_i
        ) * x_i - ratio_sigma_i * scale_i * denoised_output_i

        if self.in_first_order_step:
            self.first_order_d_i = d_i
        else:
            d_i = (d_i + self.first_order_d_i) / 2
            self.first_order_d_i = None

        x_i_next = x_i + (t_i_next - t_i) * d_i
        return x_i_next


class StochasticModifiedEulerSampler(ABC):
    def __init__(
        self,
        N: int,
        device: Union[str, torch.device],
        S_churn: float = 0.0,
        S_tmin: float = 0.0,
        S_tmax: float = float("inf"),
        S_noise: float = 1.0,
    ) -> None:
        super().__init__()
        self.N = N
        self.device = device
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        ## i [0, 0, 0, 1, 1, 1, 2, 2, 2,  ..., N-2, N-1, N-1]
        self.i_steps = torch.arange(0, N, device=self.device)
        self.i_steps = torch.cat(
            [
                self.i_steps[:-1].repeat_interleave(3),
                self.i_steps[-1:].repeat_interleave(2),
            ]
        )

        self._set_timesteps()

        self.first_order_d_i = None
        self.increased_noise_level = False

    @abstractmethod
    def _set_timesteps(self) -> None:
        """Compute self.timesteps such that t_i = self.timesteps[i]"""
        self.timesteps: torch.Tensor
        pass

    @abstractmethod
    def _get_sigma(self, i) -> torch.Tensor:
        """Compute sigma_i = sigma(t_i)"""
        return

    @abstractmethod
    def _get_scale(self, i) -> torch.Tensor:
        """Compute scale_i = scale(t_i). In the papeer scale(t_i) = s(t_i)"""
        return

    @abstractmethod
    def _get_sigma_derivative(self, i) -> torch.Tensor:
        """Compute (dsigma/dt)(t_i)"""
        return

    @abstractmethod
    def _get_scale_derivative(self, i) -> torch.Tensor:
        """Compute (dscale/dt)(t_i)"""
        return

    @property
    def in_first_order_step(self):
        return self.first_order_d_i is None

    @torch.no_grad()
    def step(
        self,
        i: Union[int, torch.Tensor],
        denoised_output_i: torch.Tensor,
        x_i: torch.Tensor,
    ):

        sigma_i = self._get_sigma(i)

        if not self.increased_noise_level:
            eps_i = torch.randn_like(x_i) * self.S_noise
            gamma_i = (
                min(self.S_churn / self.N, 2 ** (0.5) - 1)
                if self.S_tmin <= sigma_i <= self.S_tmax
                else 0.0
            )

            sigma_i_hat = (1 + gamma_i) * sigma_i
            x_i_next = x_i + (sigma_i_hat**2 - sigma_i**2) ** (0.5) * eps_i
        else:
            t_i = self.timesteps[i]
            t_i_next = self.timesteps[i + 1]

            scale_i = self._get_scale(i)
            d_sigma_i = self._get_sigma_derivative(i)
            d_scale_i = self._get_scale_derivative(i)

            ratio_sigma_i = d_sigma_i / sigma_i
            ratio_scale_i = d_scale_i / scale_i

            d_i = (
                ratio_sigma_i + ratio_scale_i
            ) * x_i - ratio_sigma_i * scale_i * denoised_output_i

            if self.in_first_order_step:
                self.first_order_d_i = d_i
            else:
                d_i = (d_i + self.first_order_d_i) / 2
                self.first_order_d_i = None

            x_i_next = x_i + (t_i_next - t_i) * d_i

        return x_i_next


class DeterministicEulerSampler(ABC):
    def __init__(
        self,
        N: int,
        device: Union[str, torch.device],
    ) -> None:
        super().__init__()
        self.N = N
        self.device = device

        # [0, 1, 2, ..., N-2, N-1]
        self.i_steps = torch.arange(0, N, device=self.device)

        self._set_timesteps()

    @abstractmethod
    def _set_timesteps(self) -> None:
        """Compute self.timesteps such that t_i = self.timesteps[i]"""
        self.timesteps: torch.Tensor
        return

    @abstractmethod
    def _get_sigma(self, i) -> torch.Tensor:
        """Compute sigma_i = sigma(t_i)"""
        return

    @abstractmethod
    def _get_scale(self, i) -> torch.Tensor:
        """Compute scale_i = scale(t_i). In the papeer scale(t_i) = s(t_i)"""
        return

    @abstractmethod
    def _get_sigma_derivative(self, i) -> torch.Tensor:
        """Compute (dsigma/dt)(t_i)"""
        return

    @abstractmethod
    def _get_scale_derivative(self, i) -> torch.Tensor:
        """Compute (dscale/dt)(t_i)"""
        return

    @torch.no_grad()
    def step(
        self,
        i: Union[int, torch.Tensor],
        denoised_output_i: torch.Tensor,
        x_i: torch.Tensor,
    ):
        t_i = self.timesteps[i]
        t_i_next = self.timesteps[i + 1]
        sigma_i = self._get_sigma(i)
        scale_i = self._get_scale(i)
        d_sigma_i = self._get_sigma_derivative(i)
        d_scale_i = self._get_scale_derivative(i)

        ratio_sigma_i = (
            d_sigma_i / sigma_i
        )  # safe: sigma_i is never 0 for the given i's
        ratio_scale_i = d_scale_i / scale_i

        d_i = (
            ratio_sigma_i + ratio_scale_i
        ) * x_i - ratio_sigma_i * scale_i * denoised_output_i

        x_i_next = x_i + (t_i_next - t_i) * d_i
        return x_i_next
