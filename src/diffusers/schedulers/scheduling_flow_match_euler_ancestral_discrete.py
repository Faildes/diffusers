# Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team.
# Licensed under the Apache License, Version 2.0

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, is_scipy_available, logging
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin

if is_scipy_available():
    import scipy.stats

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMatchEulerAncestralDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor`):
            Computed sample (x_{t-1}) of previous timestep.
        pred_original_sample (`torch.FloatTensor`, *optional*):
            Predicted x0 (helpful for preview / debugging).
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


class FlowMatchEulerAncestralDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Flow-matching Euler scheduler with Euler-Ancestral style noise injection.

    - Sigma schedule / shifting logic is inherited from FlowMatchEulerDiscreteScheduler.
    - Step update uses:
        x0 = x - sigma * v
        sigma_to = sigma_down + sigma_up decomposition (k-diffusion style)
        x_{t-1} = x + (sigma_down - sigma) * v + sigma_up * noise
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        time_shift_type: str = "exponential",
        # kept for config-compat (FlowMatchEulerDiscrete has it)
        stochastic_sampling: bool = False,
        # new: default True because this scheduler is "Ancestral"
        ancestral_sampling: bool = True,
        eta: float = 0.75,
    ):
        if self.config.use_beta_sigmas and not is_scipy_available():
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        if sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
            raise ValueError(
                "Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used."
            )
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError("`time_shift_type` must either be 'exponential' or 'linear'.")

        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self._shift = shift

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def shift(self):
        return self._shift

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def set_shift(self, shift: float):
        self._shift = shift

    @property
    def init_noise_sigma(self) -> torch.Tensor:
        # For flow-matching schedule, sigma_max is usually 1.0
        return self.sigmas.max()

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        if noise is None:
            raise ValueError("`noise` must be provided to `scale_noise`.")

        sample = sigma * noise + (1.0 - sigma) * sample
        return sample

    # Optional compatibility: some pipelines call add_noise()
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.scale_noise(original_samples, timesteps, noise)

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        if self.config.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        elif self.config.time_shift_type == "linear":
            return self._time_shift_linear(mu, sigma, t)

    def stretch_shift_to_terminal(self, t: torch.Tensor) -> torch.Tensor:
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        stretched_t = 1 - (one_minus_z / scale_factor)
        return stretched_t

    def _append_dims(self, x: torch.Tensor, target_ndim: int) -> torch.Tensor:
        while x.ndim < target_ndim:
            x = x.unsqueeze(-1)
        return x

    def _ancestral_split(self, sigma_from: torch.Tensor, sigma_to: torch.Tensor, eta: float):
        eps = 1e-20
        sigma_from_sq = sigma_from * sigma_from
        sigma_to_sq = sigma_to * sigma_to

        frac = sigma_to_sq * (sigma_from_sq - sigma_to_sq) / torch.clamp(sigma_from_sq, min=eps)
        frac = torch.clamp(frac, min=0.0)

        sigma_up = (eta * torch.sqrt(frac))
        sigma_up = torch.minimum(sigma_up, sigma_to)

        sigma_down_sq = torch.clamp(sigma_to_sq - sigma_up * sigma_up, min=0.0)
        sigma_down = torch.sqrt(sigma_down_sq)
        return sigma_down, sigma_up

    def _get_sigmas(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device.type, device.index, dtype)
        cache = getattr(self, "_sigmas_cache", None)
        if cache is None:
            cache = {}
            self._sigmas_cache = cache

        sig = cache.get(key, None)
        if sig is None or sig.shape != self.sigmas.shape:
            sig = self.sigmas.to(device=device, dtype=dtype)
            cache[key] = sig
        return sig

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        timesteps: Optional[List[float]] = None,
    ):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `True`")

        if sigmas is not None and timesteps is not None:
            if len(sigmas) != len(timesteps):
                raise ValueError("`sigmas` and `timesteps` should have the same length")

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError("`sigmas` and `timesteps` should have the same length as num_inference_steps")
        else:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)

        self.num_inference_steps = num_inference_steps

        is_timesteps_provided = timesteps is not None
        if is_timesteps_provided:
            timesteps = np.array(timesteps).astype(np.float32)

        if sigmas is None:
            if timesteps is None:
                timesteps = np.linspace(
                    self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
                )
            sigmas = timesteps / self.config.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas)

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        if self.config.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        if not is_timesteps_provided:
            timesteps = sigmas * self.config.num_train_timesteps
        else:
            timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)

        if self.config.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        else:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.timesteps = timesteps
        self.sigmas = sigmas
        self._sigmas_cache = {}
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerAncestralDiscreteSchedulerOutput, Tuple]:
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                "Passing integer indices as timesteps is not supported. "
                "Make sure to pass one of the `scheduler.timesteps`."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues
        sample_f = sample.to(torch.float32)

        # Decide whether to use ancestral behavior
        use_ancestral = bool(getattr(self.config, "ancestral_sampling", True) or getattr(self.config, "stochastic_sampling", False))

        if per_token_timesteps is not None:
            calc_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
            sample_f = sample.to(calc_dtype)
            model_out_f = model_output.to(calc_dtype)

            per_token_sigmas = (per_token_timesteps.to(device=sample.device, dtype=calc_dtype)
                                / self.config.num_train_timesteps)

            sigmas = self._get_sigmas(sample.device, calc_dtype)
            sigmas_asc = torch.flip(sigmas, dims=[0])  # 0 -> ... -> max

            per_token_sigmas_clamped = per_token_sigmas.clamp(min=sigmas_asc[0], max=sigmas_asc[-1])

            pos = torch.searchsorted(sigmas_asc, per_token_sigmas_clamped, right=False)
            idx = (pos - 1).clamp(min=0, max=sigmas_asc.numel() - 1)

            flat_idx = idx.reshape(-1)
            sigma_to = sigmas_asc.take(flat_idx).reshape(idx.shape)
            sigma_from = per_token_sigmas

            sigma_from_e = self._append_dims(sigma_from, sample_f.ndim)
            sigma_to_e   = self._append_dims(sigma_to,   sample_f.ndim)

            x0 = sample_f - sigma_from_e * model_out_f

            use_ancestral = bool(
                getattr(self.config, "ancestral_sampling", True)
                or getattr(self.config, "stochastic_sampling", False)
            )

            if not use_ancestral:
                dt = sigma_to_e - sigma_from_e
                prev_sample = sample_f + dt * model_out_f
            else:
                eta = float(getattr(self.config, "eta", 1.0))
                sigma_down, sigma_up = self._ancestral_split(sigma_from_e, sigma_to_e, eta=eta)
                dt = sigma_down - sigma_from_e
                prev_sample = sample_f + dt * model_out_f

                if torch.any(sigma_up > 0):
                    noise = randn_tensor(
                        sample_f.shape, dtype=sample_f.dtype, device=sample_f.device, generator=generator
                    )
                    prev_sample = prev_sample + noise * sigma_up * float(s_noise)

            prev_sample = prev_sample.to(model_output.dtype)

            self._step_index = (self._step_index + 1) if (self._step_index is not None) else 1

            if not return_dict:
                return (prev_sample, x0)

            return FlowMatchEulerAncestralDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=x0)


        # Scalar sigma path (standard)
        sigma_idx = self.step_index
        sigma_from = self.sigmas[sigma_idx]
        sigma_to = self.sigmas[sigma_idx + 1]
        dt_plain = sigma_to - sigma_from

        # Predicted x0 (same as FlowMatchEulerDiscrete's stochastic branch)
        x0 = sample_f - sigma_from * model_output

        if not use_ancestral:
            prev_sample = sample_f + dt_plain * model_output
        else:
            # Euler Ancestral style split
            # Guard against sigma_from == 0
            eps = 1e-12
            sigma_from_sq = sigma_from * sigma_from
            sigma_to_sq = sigma_to * sigma_to

            frac = sigma_to_sq * (sigma_from_sq - sigma_to_sq) / (sigma_from_sq + eps)
            frac = torch.clamp(frac, min=0.0)
            sigma_up = torch.sqrt(frac)

            sigma_down_sq = torch.clamp(sigma_to_sq - sigma_up * sigma_up, min=0.0)
            sigma_down = torch.sqrt(sigma_down_sq)

            dt = sigma_down - sigma_from
            prev_sample = sample_f + dt * model_output

            noise = randn_tensor(
                model_output.shape, dtype=sample_f.dtype, device=sample_f.device, generator=generator
            )
            prev_sample = prev_sample + noise * sigma_up * s_noise

        # increment step index
        self._step_index += 1

        # Cast back to model dtype (same behavior as FlowMatchEulerDiscrete)
        prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample, x0)

        return FlowMatchEulerAncestralDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=x0)

    # --- sigma schedule conversions (same as FlowMatchEulerDiscrete) ---

    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        sigma_min = getattr(self.config, "sigma_min", None)
        sigma_max = getattr(self.config, "sigma_max", None)
        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = 7.0
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def _convert_to_exponential(self, in_sigmas: torch.Tensor, num_inference_steps: int) -> torch.Tensor:
        sigma_min = getattr(self.config, "sigma_min", None)
        sigma_max = getattr(self.config, "sigma_max", None)
        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))
        return sigmas

    def _convert_to_beta(
        self, in_sigmas: torch.Tensor, num_inference_steps: int, alpha: float = 0.6, beta: float = 0.6
    ) -> torch.Tensor:
        sigma_min = getattr(self.config, "sigma_min", None)
        sigma_max = getattr(self.config, "sigma_max", None)
        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                ]
            ]
        )
        return sigmas

    def _time_shift_exponential(self, mu, sigma, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _time_shift_linear(self, mu, sigma, t):
        return mu / (mu + (1 / t - 1) ** sigma)

    def __len__(self):
        return self.config.num_train_timesteps
