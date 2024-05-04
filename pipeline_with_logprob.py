# Copied from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# with the following modifications:
# - It uses the patched version of `ddim_step_with_logprob` from `ddim_with_logprob.py`. As such, it only supports the
#   `ddim` scheduler.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.

from typing import Any, Callable, Dict, List, Optional, Union

import torch

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from ddim_with_logprob import ddim_step_with_logprob


@torch.no_grad()
def pipeline_with_logprob(
    self: StableDiffusionPipeline,
    num_inference_steps: int = 50,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    model = None,
    device = None,
):

    # 1. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    
    # 2. Prepare latent variables
    # latents = model.encode(sampling=True) # get x0
    # latents = torch.randn_like(model.encode())
    latents = model.xT
    batch_size = latents.shape[0]
        
    # 3. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    
    # 4. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    all_latents = [latents]
    all_log_probs = []
    # with self.progress_bar(total=num_inference_steps) as progress_bar:
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (latents)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        
        # predict the noise residual
        t = t.float().view(batch_size)
        noise_pred = self.unet(
            latent_model_input,
            t,
        )
        
        # compute the previous noisy sample x_t -> x_t-1
        latents, log_prob = ddim_step_with_logprob(
            self.scheduler, noise_pred, t, latents, model, **extra_step_kwargs
        )

        all_latents.append(latents)
        all_log_probs.append(log_prob)

        # call the callback, if provided
        if i == len(timesteps) - 1 or (
            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
        ):
            # progress_bar.update()
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

    image = latents
    has_nsfw_concept = None #TODO: check the meaning of nsfw

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = self.image_processor.postprocess( # return image if output_type == "latent"
        image, output_type=output_type, do_denormalize=do_denormalize # TODO: check meaning of denormalize
    )

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    return image, has_nsfw_concept, all_latents, all_log_probs