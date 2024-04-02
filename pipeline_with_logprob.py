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
    output_type: Optional[str] = "pil",
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    model = None,
    device = None,
):
    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    # self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)
    timesteps = self.scheduler.timesteps
    
    # 5. Prepare latent variables
    latents = model.encode(sampling=True) # get x0
    batch_size = latents.shape[0]
        
    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    
    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    all_latents = [latents]
    all_log_probs = []
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (latents)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # predict the noise residual
            t = t.float().view(batch_size)
            noise_pred = self.unet(latent_model_input, t)

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
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    image = latents
    has_nsfw_concept = None
    do_denormalize = [True] * image.shape[0]

    image = self.image_processor.postprocess( #TODO: check this process detail
        image, output_type=output_type, do_denormalize=do_denormalize
    )

    # Offload last model to CPU TODO: what's this?
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    return image, has_nsfw_concept, all_latents, all_log_probs
