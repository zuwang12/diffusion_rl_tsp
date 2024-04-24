import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from functools import partial
import tqdm
import numpy as np
import pandas as pd
import gc
import torch
from torchvision.utils import save_image

from model.unet import UNetModel
from model.TSPModel import Model_x0, TSPDataset

from diffusers import StableDiffusionPipeline, DDIMScheduler

import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob

from utils import TSP_2opt

seed = 2024
deterministic = True

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

device = ('cuda' if torch.cuda.is_available() else 'cpu')

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    config.file_name = f'tsp{config.num_cities}_test_concorde.txt'
    config.result_file_name = f'ours_tsp{config.num_cities}_test_epoch{config.num_epochs}_inner{config.train.num_inner_epochs}_{config.run_name}.csv'
    
    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )
    
    ########### add prior Unet ###########
    unet = UNetModel(image_size=config.image.img_size, in_channels=1, out_channels=1, 
                                model_channels=64, num_res_blocks=2, channel_mult=(1,2,3,4),
                                attention_resolutions=[16,8], num_heads=4).to(device)
    unet.load_state_dict(torch.load(f'Checkpoint/unet50_64_8.pth'))
    unet.to(device)

    pipeline.unet = unet
    print('Loaded model')
    ########### add prior Unet ###########
    
    # freeze parameters of models to save more memory
    del pipeline.vae, pipeline.text_encoder
    pipeline.unet.requires_grad_(False)
    pipeline.safety_checker = None

    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    unet = pipeline.unet

    # Enable TF32 for faster training on Ampere GPUs
    if config.allow_tf32: # TODO: need to change
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer_cls = torch.optim.AdamW
        
    ############ add dataloader ###########
    test_dataset = TSPDataset(data_file=f'./data/{config.file_name}',
                              img_size = config.image.img_size,
                              point_radius = config.image.point_radius,
                              point_color = config.image.point_color,
                              point_circle = config.image.point_circle,
                              line_thickness = config.image.line_thickness,
                              line_color = config.image.line_color)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.sample.batch_size, shuffle=False)
    num_points = test_dataset.rasterize(0)[1].shape[0]
    print('Created dataset')
    
    solved_costs = []
    init_costs = []
    gt_costs = []
    final_gaps = []
    init_gaps = []
    epochs = []
    
    for (img, sample_idx) in tqdm(test_dataloader):
        ########### add prior model & prepare image ###########
        model = Model_x0( # TODO: From define -> To reinit
            batch_size = config.sample.batch_size, 
            num_points = num_points, 
            img_size = config.image.img_size,
            line_color = config.image.line_color,
            line_thickness = config.image.line_thickness,).to(device) #TODO: check batch_size from sample vs train
        
        ########### add prior model & prepare image ###########
        _, points, gt_tour = test_dataset.rasterize(sample_idx[0].item())
        solver = TSP_2opt(points)
        gt_cost = solver.evaluate([i-1 for i in gt_tour])
        img_query = torch.zeros_like(img)
        img_query[img == 1] = 1
        model.compute_edge_images(points=points, img_query=img_query)
        
        dists = np.zeros_like(model.latent[0].cpu().detach()) # (50, 50)
        for i in range(dists.shape[0]):
            for j in range(dists.shape[0]):
                dists[i,j] = np.linalg.norm(points[i]-points[j])
        
        ########### add prior model & prepare image ###########
        
        optimizer = optimizer_cls(
            model.parameters(), # unet.parameters()
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        )

        reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)()
 
        # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
        # remote server running llava inference. TODO: llava 안쓰면?
        executor = futures.ThreadPoolExecutor(max_workers=2)

        # Train!
        samples_per_epoch = (
            config.sample.batch_size * config.sample.num_batches_per_epoch
        )
        total_train_batch_size = (
            config.train.batch_size * config.train.gradient_accumulation_steps
        )

        assert config.sample.batch_size >= config.train.batch_size
        assert config.sample.batch_size % config.train.batch_size == 0
        assert samples_per_epoch % total_train_batch_size == 0

        first_epoch = 0
        
        final_solved_cost = 10**10
        final_gap = 0
        
        pipeline.unet.eval()
        for epoch in range(first_epoch, config.num_epochs):
            #################### SAMPLING ####################
            samples = []
            images, _, latents, log_probs = pipeline_with_logprob( # TODO: need to be modified
                pipeline,
                num_inference_steps=config.sample.num_steps,
                eta=config.sample.eta,
                output_type="latent", # output_type="pt"
                model = model,
                device = device
            )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, points, model.latent, dists)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

            # wait for all rewards to be computed
            for sample in samples:
                rewards, reward_metadata = sample["rewards"].result()
                sample["rewards"] = torch.as_tensor(rewards, device=device)

            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
            
            # gather rewards across processes
            rewards = (samples["rewards"]).cpu().numpy()
            solved_cost = reward_metadata['solved_cost']
            gap = 100*(solved_cost-gt_cost) / gt_cost
            
            if solved_cost<final_solved_cost:
                running_epoch = epoch
                final_solved_cost = solved_cost
                final_gap = gap

            ################################## saving Init Cost ##################################
            if epoch == 0:
                init_cost = solved_cost
                init_gap = gap
            ################################## saving Init Cost ##################################
            
            # advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            advantages = rewards

            # ungather advantages; we only need to keep the entries corresponding to the samples on this process
            samples["advantages"] = (
                torch.as_tensor(advantages)
                .reshape(1, -1)[0]
                .to(device)
            )

            del samples["rewards"]

            total_batch_size, num_timesteps = samples["timesteps"].shape
            assert (
                total_batch_size
                == config.sample.batch_size * config.sample.num_batches_per_epoch
            )
            assert num_timesteps == config.sample.num_steps

            #################### TRAINING ####################
            for inner_epoch in range(config.train.num_inner_epochs):
                # shuffle samples along batch dimension
                perm = torch.randperm(total_batch_size, device=device)
                samples = {k: v[perm] for k, v in samples.items()}

                # shuffle along time dimension independently for each sample
                perms = torch.stack(
                    [
                        torch.randperm(num_timesteps, device=device)
                        for _ in range(total_batch_size)
                    ]
                )
                for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                    samples[key] = samples[key][
                        torch.arange(total_batch_size, device=device)[:, None],
                        perms,
                    ]

                # rebatch for training
                samples_batched = {
                    k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
                    for k, v in samples.items()
                }

                # dict of lists -> list of dicts for easier iteration
                samples_batched = [
                    dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
                ]

                # train
                model.train()
                for i, sample in list(enumerate(samples_batched)):
                    for j in range(num_train_timesteps):
                        noise_pred = unet(
                            sample["latents"][:, j],
                            sample["timesteps"][:, j],
                        )
                        
                        # compute the log prob of next_latents given latents under the current model
                        _, log_prob = ddim_step_with_logprob(
                            pipeline.scheduler,
                            noise_pred,
                            sample["timesteps"][:, j],
                            sample["latents"][:, j],
                            model,
                            eta=config.sample.eta,
                            prev_sample=sample["next_latents"][:, j],
                        )

                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # backward pass
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

        ################################## saving Pred ##################################
        solved_costs.append(final_solved_cost)
        init_costs.append(init_cost)
        gt_costs.append(gt_cost)
        final_gaps.append(final_gap)
        init_gaps.append(init_gap)
        epochs.append(running_epoch)
        
        del loss
        gc.collect()
        torch.cuda.empty_cache()
        
        if sample_idx%config.save_freq==1:
            result_df = pd.DataFrame({
                'solved_cost' : solved_costs,
                'init_cost' : init_costs,
                'gt_cost' : gt_costs,
                'final_gap(%)' : final_gaps,
                'init_gap(%)' : init_gaps,
                'best_epoch' : epochs,
            })
            if config.save_result:
                result_df.to_csv(f'./Results/{config.result_file_name}', index=False)
    else:        
        result_df = pd.DataFrame({
            'solved_cost' : solved_costs,
            'init_cost' : init_costs,
            'gt_cost' : gt_costs,
            'final_gap(%)' : final_gaps,
            'init_gap(%)' : init_gaps,
        })
        if config.save_result:
            result_df.to_csv(f'./Results/{config.result_file_name}', index=False)

if __name__ == "__main__":
    app.run(main)
