from diffusers import StableDiffusionPipeline, DDIMScheduler

import torch
from torchvision.utils import save_image

import argparse
import yaml
from munch import munchify
import numpy as np
import tqdm
from functools import partial
import time
from concurrent import futures
import pandas as pd
import gc

from model.unet import UNetModel
from model.TSPModel import Model_x0, TSPDataset
from utils import TSP_2opt, runlat

import reward_fns
from pipeline_with_logprob import pipeline_with_logprob
from ddim_with_logprob import ddim_step_with_logprob

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="tsp", help="which training config to use")
    args = parser.parse_args()
    # mapping from config name to config path
    config_mapping = {"tsp":  "./configs/train_configs.yaml"}
    with open(config_mapping[args.config_name]) as file:
        config_dict= yaml.safe_load(file)
        config = munchify(config_dict)
    return config


if __name__=='__main__':
    config = load_config()
    tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    config.file_name = f'tsp{config.num_cities}_test_concorde.txt'
    config.result_file_name = f'ours_tsp{config.num_cities}_test_epoch{config.num_epochs}_inner{config.num_inner_epochs}_{config.run_name}.csv'
    print(config)
    
    ################## fix seed ####################
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    ################## fix seed ####################
    
    num_train_timesteps = int(config.num_steps * config.timestep_fraction)
    pipeline = StableDiffusionPipeline.from_pretrained(config.model, revision=config.revision)
    
    ################## Set model, scheduler ##################
    unet = UNetModel(image_size=config.img_size, in_channels=1, out_channels=1, model_channels=64, num_res_blocks=2, channel_mult=(1,2,3,4), attention_resolutions=[16,8], num_heads=4).to(device)
    unet.load_state_dict(torch.load(f'./ckpt/unet50_64_8.pth'))
    unet.to(device)
    unet.eval()
    pipeline.unet = unet
    print('Loaded model')
    
    del pipeline.vae, pipeline.text_encoder
    # pipeline.unet.requires_grad_(False)
    pipeline.safety_checker = None

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # Enable TF32 for faster training on Ampere GPUs
    if config.allow_tf32: # TODO: need to change
        torch.backends.cuda.matmul.allow_tf32 = True
    ################## Set model, scheduler ##################
    
    optimizer_cls = torch.optim.AdamW
        
    test_dataset = TSPDataset(data_file=f'./data/{config.file_name}',
                              img_size = config.img_size,
                              point_radius = config.point_radius,
                              point_color = config.point_color,
                              point_circle = config.point_circle,
                              line_thickness = config.line_thickness,
                              line_color = config.line_color)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size_sample, shuffle=False)
    num_points = test_dataset.rasterize(0)[1].shape[0]
    print('Created dataset')
    
    solved_costs, init_costs, gt_costs, final_gaps, init_gaps, epochs = [], [], [], [], [], []

    for (img, points, gt_tour, sample_idx) in tqdm(test_dataloader):
        ########### add prior model & prepare image ###########
        model = Model_x0( # TODO: From define -> To reinit
            batch_size = config.batch_size_sample, 
            num_points = num_points, 
            img_size = config.img_size,
            line_color = config.line_color,
            line_thickness = config.line_thickness,).to(device) #TODO: check batch_size from sample vs train

        # _, points, gt_tour = test_dataset.rasterize(sample_idx[0].item())
        points, gt_tour = points.numpy()[0], gt_tour.numpy()[0]
        solver = TSP_2opt(points)
        gt_cost = solver.evaluate([i-1 for i in gt_tour])
        img_query = torch.zeros_like(img)
        img_query[img == 1] = 1
        model.compute_edge_images(points=points, img_query=img_query) # Pre-compute edge images
        
        dists = np.zeros_like(model.latent[0].cpu().detach()) # (50, 50)
        for i in range(dists.shape[0]):
            for j in range(dists.shape[0]):
                dists[i,j] = np.linalg.norm(points[i]-points[j])
                
        if config.use_prior_init:
            runlat(model, unet, STEPS=config.num_steps, batch_size=1, device=device)
        ########### add prior model & prepare image ###########
        
        optimizer = optimizer_cls(
            model.parameters(), # unet.parameters()
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon
            )
        
        reward_fn = getattr(reward_fns, config.reward_type)()
        executor = futures.ThreadPoolExecutor(max_workers=2)

        # Train!
        # samples_per_epoch = (config.batch_size_sample * config.num_processes * config.num_batches_per_epoch) # what the hell is this?
        # total_train_batch_size = (config.batch_size_train * config.num_processes * config.gradient_accumulation_steps) # what the hell is this?
        
        final_solved_cost = 10**10
        final_gap = 0
        for epoch in range(config.num_epochs):
            #################### SAMPLING ####################
            samples = []
            for i in range(config.num_batches_per_epoch):
                images, _, latents, log_probs = pipeline_with_logprob(
                    pipeline,
                    num_inference_steps = config.num_steps,
                    eta=config.eta,
                    output_type="latent", # output_type="pt"
                    model = model,
                    device = device
                )

                latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
                log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
                timesteps = pipeline.scheduler.timesteps.repeat(config.batch_size_sample, 1)  # (batch_size, num_steps)

                # compute rewards asynchronously
                rewards = executor.submit(reward_fn, points, model.latent, dists)
                # yield to to make sure reward computation starts
                time.sleep(0)

                samples.append(
                    {
                        "timesteps": timesteps,
                        "latents": latents[:, :-1],  # each entry is the latent before timestep t
                        "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
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
            samples["advantages"] = (torch.as_tensor(advantages).reshape(config.num_processes, -1)[config.process_index].to(device))
            del samples["rewards"]

            total_batch_size, num_timesteps = samples["timesteps"].shape

            #################### TRAINING ####################
            for inner_epoch in tqdm(range(config.num_inner_epochs)):
                # shuffle samples along batch dimension
                perm = torch.randperm(total_batch_size, device=device)
                samples = {k: v[perm] for k, v in samples.items()}

                # shuffle along time dimension independently for each sample
                perms = torch.stack([torch.randperm(num_timesteps, device=device) for _ in range(total_batch_size)])
                for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                    samples[key] = samples[key][torch.arange(total_batch_size, device=device)[:, None], perms, ]

                # rebatch for training
                samples_batched = {k: v.reshape(-1, config.batch_size_train, *v.shape[1:]) for k, v in samples.items()}

                # dict of lists -> list of dicts for easier iteration
                samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

                for i, sample in list(enumerate(samples_batched)):
                    for j in range(num_train_timesteps):
                        noise_pred = unet(sample["latents"][:, j], sample["timesteps"][:, j],)
                        
                        _, log_prob = ddim_step_with_logprob(
                            pipeline.scheduler,
                            noise_pred,
                            sample["timesteps"][:, j],
                            sample["latents"][:, j],
                            model,
                            eta=config.eta,
                            prev_sample=sample["next_latents"][:, j],
                        )

                        # ppo logic
                        advantages = torch.clamp(sample["advantages"], -config.adv_clip_max, config.adv_clip_max,)
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range, )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

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