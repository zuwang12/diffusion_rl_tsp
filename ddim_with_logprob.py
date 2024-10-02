from typing import Optional, Tuple, Union

import math
import torch

from diffusers.utils import randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu().type(torch.int64)).to(
        timestep.device
    )
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu().type(torch.int64)),
        self.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance


def ddim_step_with_logprob(
    self: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    model,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> Union[DDIMSchedulerOutput, Tuple]:
    assert isinstance(self, DDIMScheduler)
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = (
        timestep - self.config.num_train_timesteps // self.num_inference_steps
    )
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu().type(torch.int64))
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu().type(torch.int64)),
        self.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(
        sample.device
    )

    beta_prod_t = 1 - alpha_prod_t
        
    pred_original_sample = model.encode()
    pred_epsilon = model_output

    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    variance = _get_variance(self, timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    if use_clipped_model_output:
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)

    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * pred_epsilon

    prev_sample_mean = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.type(sample.dtype), log_prob
