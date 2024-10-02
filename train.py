import json
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import argparse
import yaml
from munch import munchify
import numpy as np
import tqdm
from functools import partial
import time
import pandas as pd
import gc
import os
from copy import deepcopy
from model.unet import UNetModel
from model.TSPModel import Model_x0, TSPDataset
from utils import TSP_2opt, runlat, calculate_distance_matrix2
import reward_fns
from pipeline_with_logprob import pipeline_with_logprob
from ddim_with_logprob import ddim_step_with_logprob
from diffusers import StableDiffusionPipeline, DDIMScheduler

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="tsp", help="which training config to use")
    parser.add_argument("--start_idx", type=int, default=0, help="start index for iteration")
    parser.add_argument("--end_idx", type=int, default=1280, help="end index for iteration")
    parser.add_argument("--num_cities", type=int, default=200, help="number of cities")
    parser.add_argument("--max_iter", type=int, default=5, help="max iteration of 2opt at N=200")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of epoch")
    parser.add_argument("--num_inner_epochs", type=int, default=1, help="number of inner epoch")
    parser.add_argument("--num_init_sample", type=int, default=1, help="number of initial sample")
    parser.add_argument("--run_name", type=str, default='tsp_test', help="Name for the run")
    parser.add_argument("--constraint_type", type=str, default='path')
    args = parser.parse_args()
    config_mapping = {"tsp": "./configs/train_configs.yaml"}
    with open(config_mapping[args.config_name]) as file:
        config_dict = yaml.safe_load(file)
        config = munchify(config_dict)

    config.start_idx = args.start_idx
    config.end_idx = args.end_idx
    config.num_cities = args.num_cities
    config.max_iter = args.max_iter
    config.run_name = args.run_name
    config.num_epochs = args.num_epochs
    config.num_inner_epochs = args.num_inner_epochs
    config.num_init_sample = args.num_init_sample
    config.constraint_type = args.constraint_type
    return config

def main():
    # accelerator = Accelerator(mixed_precision='fp16')  # Enable mixed precision with fp16
    date_per_type = {
        'basic': '',
        'box': '240710',
        'path': '240711',
        'cluster': '240721',
    }
    config = load_config()
    tqdm_partial = partial(tqdm.tqdm, dynamic_ncols=True)
    now = time.strftime('%y%m%d_%H%M%S')
    if config.run_name is None:
        config.run_name = f'test_{now}'
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    # device = accelerator.device
    # dtype = torch.float16 if config.use_fp16 else torch.float32

    if config.constraint_type == 'basic':
        config.file_name = f'tsp{config.num_cities}_test_concorde.txt'
        config.result_file_name = f'ours_tsp{config.num_cities}_{config.constraint_type}_{config.start_idx}_{config.end_idx}.csv'
    else:
        config.file_name = f'tsp{config.num_cities}_{config.constraint_type}_constraint_{date_per_type.get(config.constraint_type)}.txt'
        config.result_file_name = f'ours_tsp{config.num_cities}_{config.constraint_type}_constraint_{config.start_idx}_{config.end_idx}.csv'
    print(json.dumps(config, indent=4))
    print(f'Result file : ./Results/{config.constraint_type}/{config.run_name}/{config.result_file_name}')
    
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    num_train_timesteps = int(config.num_steps * config.timestep_fraction)
    pipeline = StableDiffusionPipeline.from_pretrained(config.model, revision=config.revision)
    unet = UNetModel(
        image_size=config.img_size, 
        in_channels=1, 
        out_channels=1, 
        model_channels=64,
        num_res_blocks=2, 
        channel_mult=(1, 2, 3, 4), 
        attention_resolutions=[16, 8], 
        num_heads=4, 
        use_fp16=config.use_fp16
    ).to(device)
    
    if config.use_prior:
        unet.load_state_dict(torch.load(f'./ckpt/unet50_64_8.pth', map_location=device))
    unet.eval()
    pipeline.unet = unet
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    del pipeline.vae, pipeline.text_encoder
    pipeline.safety_checker = None

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    test_dataset = TSPDataset(
        data_file=f'./data/{config.file_name}',
        img_size=config.img_size,
        constraint_type=config.constraint_type,
        point_radius=config.point_radius,
        point_color=config.point_color,
        point_circle=config.point_circle,
        line_thickness=config.line_thickness,
        line_color=config.line_color,
        show_position=False
    )

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size_sample, 
        shuffle=False,
        pin_memory=True,  # Use pin_memory to speed up data transfer to GPU
        num_workers=16     # Adjust based on your CPU capability
    )
    # test_dataloader = accelerator.prepare(test_dataloader)

    sample_idxes, solved_costs, gt_costs, final_gaps, epochs, inner_epochs, basic_costs, penalty_counts = [], [], [], [], [], [], [], []

    for img, points, gt_tour, sample_idx, constraint in tqdm_partial(test_dataloader):
        points, gt_tour = points.cpu().numpy()[0], gt_tour.cpu().numpy()[0]
        if config.constraint_type != 'basic':
            constraint = constraint.cpu().numpy()[0]
        else:
            constraint = None
            
        if not (config.start_idx <= int(sample_idx) < config.end_idx):
            continue

        xT = torch.randn_like(img).float().to(device)
        model = Model_x0(
            batch_size=config.batch_size_sample,
            num_points=points.shape[0],
            img_size=config.img_size,
            line_color=config.line_color,
            line_thickness=config.line_thickness,
            xT=xT
        ).to(device)
        model.eval()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon
        )
        # model, optimizer = accelerator.prepare(model, optimizer)
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=num_inner_epochs)

        if config.constraint_type == 'box':
            _, intersection_matrix = calculate_distance_matrix2(points, constraint)
            box = constraint
            constraint = intersection_matrix

        solver = TSP_2opt(points, constraint_type=config.constraint_type, constraint=constraint)
        gt_cost = solver.evaluate([i - 1 for i in gt_tour])
        img_query = torch.zeros_like(img)
        img_query[img == 1] = 1
        model.compute_edge_images(points=points, img_query=img_query)

        dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

        reward_fn = getattr(reward_fns, config.reward_type)()
        final_solved_cost = 10**10
        final_gap = 0

        for epoch in range(config.num_epochs):
            num_sample = config.num_init_sample if epoch == 0 else 1
            samples = []
            best_reward = -10**10
            for _ in range(num_sample):
                if epoch == 0:
                    model.reset()
                    if config.use_plug_and_play:
                        runlat(model, unet, STEPS=config.num_steps, batch_size=1, device=device)

                # with accelerator.autocast():
                _, _, latents, log_probs = pipeline_with_logprob(
                    pipeline,
                    num_inference_steps=config.num_steps,
                    eta=config.eta,
                    output_type="latent",
                    model=model,
                    device=device
                )

                latents = torch.stack(latents, dim=1)
                log_probs = torch.stack(log_probs, dim=1)
                timesteps = pipeline.scheduler.timesteps.repeat(config.batch_size_sample, 1)

                if config.constraint_type == 'box':
                    constraint = intersection_matrix
                rewards = torch.as_tensor(
                    reward_fn(points, model.latent, dists, config.constraint_type, constraint, config.max_iter)[0],
                    device=device
                )

                if config.use_best_sample and float(rewards) > best_reward:
                    best_reward = float(rewards)
                    model_init = deepcopy(model.state_dict())

                samples.append({
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": log_probs,
                    "rewards": rewards,
                })

            if epoch == 0:
                if config.use_best_sample:
                    model.load_state_dict(model_init)
                else:
                    model.reset()

            samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
            rewards = samples["rewards"].cpu().numpy()
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8) if len(rewards) > 1 else rewards
            samples["advantages"] = torch.as_tensor(advantages).reshape(config.num_processes, -1)[config.process_index].to(device)
            del samples["rewards"]

            total_batch_size, num_timesteps = samples["timesteps"].shape
            num_inner_epochs = config.num_inner_epochs if epoch == 0 else 1

            for inner_epoch in range(num_inner_epochs):
                perm = torch.randperm(total_batch_size, device=device)
                samples = {k: v[perm] for k, v in samples.items()}

                perms = torch.stack([torch.randperm(num_timesteps, device=device) for _ in range(total_batch_size)])
                for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                    samples[key] = samples[key][torch.arange(total_batch_size, device=device)[:, None], perms]

                samples_batched = {k: v.reshape(-1, config.batch_size_train, *v.shape[1:]) for k, v in samples.items()}
                samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

                for sample in samples_batched:
                    for j in range(num_train_timesteps):
                        # with accelerator.autocast():
                        noise_pred = unet(sample["latents"][:, j], sample["timesteps"][:, j])
                        _, log_prob = ddim_step_with_logprob(
                            pipeline.scheduler,
                            noise_pred,
                            sample["timesteps"][:, j],
                            sample["latents"][:, j],
                            model,
                            eta=config.eta,
                            prev_sample=sample["next_latents"][:, j],
                        )

                        advantages = torch.clamp(sample["advantages"], -config.adv_clip_max, config.adv_clip_max)
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        optimizer.zero_grad()
                        loss.backward()
                        # accelerator.backward(loss)
                        optimizer.step()
                    output = reward_fn(points, model.latent, dists, config.constraint_type, constraint=constraint, max_iter = config.max_iter)[1]

                    solved_tour, basic_cost, penalty_count = output['solved_tour'], output['basic_cost'], output['penalty_count']
                    solved_cost = basic_cost + config.penalty_const * penalty_count 
                    gap = 100 * (solved_cost - gt_cost) / gt_cost
                    if solved_cost < final_solved_cost:
                        best_epoch = epoch
                        best_inner_epoch = inner_epoch
                        final_solved_cost = solved_cost
                        final_gap = gap
                        final_basic_cost = basic_cost
                        final_penalty_count = penalty_count

        sample_idxes.append(int(sample_idx))
        solved_costs.append(final_solved_cost)
        gt_costs.append(gt_cost)
        final_gaps.append(final_gap)
        epochs.append(best_epoch)
        inner_epochs.append(best_inner_epoch)
        basic_costs.append(final_basic_cost)
        penalty_counts.append(final_penalty_count)

        del loss
        gc.collect()
        torch.cuda.empty_cache()

        if sample_idx % config.save_freq == 0:
            result_df = pd.DataFrame({
                'sample_idx': sample_idxes,
                'best_epoch': epochs,
                'best_inner_epoch': inner_epochs,
                'solved_cost': solved_costs,
                'gt_cost': gt_costs,
                'basic_cost': basic_costs,
                'penalty_count': penalty_counts,
                'final_gap(%)': final_gaps,
            })
            if config.save_result:
                os.makedirs(f'./Results/{config.constraint_type}/{config.run_name}', exist_ok=True)
                result_df.to_csv(f'./Results/{config.constraint_type}/{config.run_name}/{config.result_file_name}', index=False)

    else:
        result_df = pd.DataFrame({
            'sample_idx': sample_idxes,
            'best_epoch': epochs,
            'best_inner_epoch': inner_epochs,
            'solved_cost': solved_costs,
            'gt_cost': gt_costs,
            'basic_cost': basic_costs,
            'penalty_count': penalty_counts,
            'final_gap(%)': final_gaps,
        })
        if config.save_result:
            print('save result')
            os.makedirs(f'./Results/{config.constraint_type}/{config.run_name}', exist_ok=True)
            result_df.to_csv(f'./Results/{config.constraint_type}/{config.run_name}/{config.result_file_name}', index=False)

if __name__ == '__main__':
    main()
