import json
import torch
from torchvision.utils import save_image
from matplotlib import pyplot as plt

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
import os
from copy import deepcopy

from model.unet import UNetModel
from model.TSPModel import Model_x0, TSPDataset
from utils import TSP_2opt, runlat, calculate_distance_matrix2, save_figure

import reward_fns
from pipeline_with_logprob import pipeline_with_logprob
from ddim_with_logprob import ddim_step_with_logprob
from diffusers import StableDiffusionPipeline, DDIMScheduler

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="tsp", help="which training config to use")
    parser.add_argument("--start_idx", type=int, default=0, help="start index for iteration")
    parser.add_argument("--end_idx", type=int, default=1280, help="end index for iteration")
    parser.add_argument("--num_cities", type=int, default=20, help="number of cities")
    parser.add_argument("--num_epochs", type=int, default=3, help="number of epoch")
    parser.add_argument("--num_inner_epochs", type=int, default=3, help="number of inner epoch")
    parser.add_argument("--num_init_sample", type=int, default=3, help="number of initial sample")
    parser.add_argument("--run_name", type=str, default='tsp_test', help="Name for the run")
    parser.add_argument("--constraint_type", type=str, default='basic')
    args = parser.parse_args()
    # mapping from config name to config path
    config_mapping = {"tsp":  "./configs/train_configs.yaml"}
    with open(config_mapping[args.config_name]) as file:
        config_dict= yaml.safe_load(file)
        config = munchify(config_dict)
    
    # Add start_idx, end_idx, and gpu_id to config
    config.start_idx = args.start_idx
    config.end_idx = args.end_idx
    config.num_cities = args.num_cities
    config.run_name = args.run_name
    config.num_epochs = args.num_epochs
    config.num_inner_epochs = args.num_inner_epochs
    config.num_init_sample = args.num_init_sample
    config.constraint_type = args.constraint_type
    return config

if __name__=='__main__':
    date_per_type = {
       'basic' : '',
       'box' : '240710',
       'path' : '240711',
       'cluster' : '240721', 
    }
    config = load_config()
    tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
    now = time.strftime('%y%m%d_%H%M%S')
    if config.run_name==None:
        config.run_name = f'test_{now}'
    device = f'cuda' if torch.cuda.is_available() else 'cpu'

    if config.constraint_type == 'basic':
        config.file_name = f'tsp{config.num_cities}_test_concorde.txt'
        config.result_file_name = f'ours_tsp{config.num_cities}_{config.constraint_type}_{config.start_idx}_{config.end_idx}.csv'
    else:
        config.file_name = f'tsp{config.num_cities}_{config.constraint_type}_constraint_{date_per_type.get(config.constraint_type)}.txt'
        config.result_file_name = f'ours_tsp{config.num_cities}_{config.constraint_type}_constraint_{config.start_idx}_{config.end_idx}.csv'
    print(json.dumps(config, indent=4))
    print(f'Result file : ./Results/{config.constraint_type}/{config.run_name}/{config.result_file_name}')
    
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
    
    ################## Set model, diffusion scheduler ##################
    unet = UNetModel(image_size=config.img_size, in_channels=1, out_channels=1, model_channels=64, num_res_blocks=2, channel_mult=(1,2,3,4), attention_resolutions=[16,8], num_heads=4).to(device)
    if config.use_prior:
        unet.load_state_dict(torch.load(f'./ckpt/unet50_64_8.pth', map_location=device))
    unet.to(device)
    unet.eval()
    pipeline.unet = unet
    print('Loaded model')
    
    del pipeline.vae, pipeline.text_encoder
    pipeline.safety_checker = None

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # Enable TF32 for faster training on Ampere GPUs
    if config.allow_tf32: 
        torch.backends.cuda.matmul.allow_tf32 = True
    ################## Set model, scheduler ##################
        
    test_dataset = TSPDataset(data_file=f'./data/{config.file_name}',
                              img_size = config.img_size,
                              constraint_type= config.constraint_type,
                              point_radius = config.point_radius,
                              point_color = config.point_color,
                              point_circle = config.point_circle,
                              line_thickness = config.line_thickness,
                              line_color = config.line_color,
                              show_position=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size_sample, shuffle=False)
    num_points = test_dataset.rasterize(0)[1].shape[0]
    print('Created dataset')
    
    sample_idxes, solved_costs, init_costs, gt_costs, final_gaps, init_gaps, epochs, inner_epochs, basic_costs, penalty_counts = [], [], [], [], [], [], [], [], [], []

    for img, points, gt_tour, sample_idx, constraint in tqdm(test_dataloader):
        # if int(sample_idx)!= 160:continue
        if config.constraint_type == 'basic':
            constraint = None
        if not (config.start_idx <= int(sample_idx) < config.end_idx):
            continue

        ########### prepare constraint ##############
        if config.constraint_type == 'box':
            distance_matrix, intersection_matrix = calculate_distance_matrix2(points[0], constraint[0])
            
        ########### add prior model & prepare image ###########
        xT = torch.randn_like(img).float().to(device) 
        model = Model_x0(
            batch_size = config.batch_size_sample,
            num_points = num_points,
            img_size = config.img_size,
            line_color = config.line_color,
            line_thickness = config.line_thickness,
            xT = xT).to(device)
        model.eval()

        if config.constraint_type == 'basic':
            points, gt_tour = points.numpy()[0], gt_tour.numpy()[0]
        else:
            points, gt_tour, constraint = points.numpy()[0], gt_tour.numpy()[0], constraint.numpy()[0]
            
        if config.constraint_type == 'box':
            constraint = intersection_matrix
            
        solver = TSP_2opt(points, constraint_type=config.constraint_type, constraint=constraint)
        gt_cost = solver.evaluate([i-1 for i in gt_tour])
        img_query = torch.zeros_like(img)
        img_query[img == 1] = 1
        model.compute_edge_images(points=points, img_query=img_query) 
        
        dists = np.zeros_like(model.latent[0].cpu().detach())
        for i in range(dists.shape[0]):
            for j in range(dists.shape[0]):
                dists[i,j] = np.linalg.norm(points[i]-points[j])

        # if config.constraint_type == 'box':
        #     gt_img = test_dataset.draw_tour(tour = gt_tour, points = points, box = constraint)
        # if config.constraint_type == 'path':
        #     gt_img = test_dataset.draw_tour(tour = gt_tour, points = points, paths = constraint)
        # if config.constraint_type == 'cluster':
        #     gt_img = test_dataset.draw_tour(tour = gt_tour, points = points, cluster = constraint)
        
        reward_fn = getattr(reward_fns, config.reward_type)()
        final_solved_cost = 10**10
        final_gap = 0
        
        for epoch in range(config.num_epochs):           
            if epoch == 0:
                num_sample = config.num_init_sample
            else:
                num_sample = 1 # TODO: check this approach
            samples = []
            best_reward = -10**10
            for i in range(num_sample):
                if epoch == 0:
                    model.reset() 
                    if config.use_plug_and_play:
                        runlat(model, unet, STEPS=config.num_steps, batch_size=1, device=device)   
                _, _, latents, log_probs = pipeline_with_logprob(
                    pipeline,
                    num_inference_steps = config.num_steps,
                    eta=config.eta,
                    output_type="latent", 
                    model = model,
                    device = device
                )

                latents = torch.stack(latents, dim=1)  
                log_probs = torch.stack(log_probs, dim=1)  
                timesteps = pipeline.scheduler.timesteps.repeat(config.batch_size_sample, 1)
                
                if config.constraint_type == 'box':
                    constraint = intersection_matrix
                rewards = torch.as_tensor(reward_fn(points, model.latent, dists, config.constraint_type, constraint)[0], device=device)
                # if config.constraint_type == 'box':
                #     rewards = torch.as_tensor(reward_fn(points, model.latent, dists, config.constraint_type, constraint)[0], device=device)
                # elif config.constraint_type == 'path':
                #     rewards = torch.as_tensor(reward_fn(points, model.latent, dists, path = constraint)[0], device=device)
                # elif config.constraint_type == 'cluster':
                #     rewards = torch.as_tensor(reward_fn(points, model.latent, dists, cluster = constraint)[0], device=device)
                    
                if config.use_best_sample & (float(rewards)>best_reward):
                    best_reward = float(rewards)
                    model_init = deepcopy(model.state_dict())

                samples.append(
                    {
                        "timesteps": timesteps,
                        "latents": latents[:, :-1],  
                        "next_latents": latents[:, 1:], 
                        "log_probs": log_probs,
                        "rewards": rewards,
                    }
                ) 
            if epoch==0:
                if config.use_best_sample:
                    model.load_state_dict(model_init)
                else:
                    model.reset()

            samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
            
            rewards = (samples["rewards"]).cpu().numpy()
            
            solved_cost = -best_reward
            gap = 100*(solved_cost-gt_cost) / gt_cost
            
            if epoch == 0:
                init_cost = solved_cost
                init_gap = gap

            if len(rewards)>1:
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                advantages = rewards

            samples["advantages"] = (torch.as_tensor(advantages).reshape(config.num_processes, -1)[config.process_index].to(device))
            del samples["rewards"]

            total_batch_size, num_timesteps = samples["timesteps"].shape
            
            if epoch == 0:
                num_inner_epochs = config.num_inner_epochs
            else:
                num_inner_epochs = 1
            
            optimizer_cls = torch.optim.AdamW
            optimizer = optimizer_cls(
                model.parameters(),
                lr =config.learning_rate,
                betas = (config.adam_beta1, config.adam_beta2),
                weight_decay = config.adam_weight_decay,
                eps = config.adam_epsilon
            )    
            
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=num_inner_epochs)
            # if epoch == 0:
            #     for i in [0, 10, 20, 30, 40, 49]:
            #         save_figure(img = samples['latents'][0,i,0,:,:].detach().cpu().numpy(), path = f'./images/encoding/xt_idx{int(sample_idx)}_timestep{i}.png')
            for inner_epoch in range(num_inner_epochs):
                # save_figure(img = model.encode()[0,0,:,:].detach().cpu().numpy(), path = f'./images/encoding/encoding_idx{int(sample_idx)}_epoch{epoch}_inner{inner_epoch}.png')
                
                perm = torch.randperm(total_batch_size, device=device)
                samples = {k: v[perm] for k, v in samples.items()}

                perms = torch.stack([torch.randperm(num_timesteps, device=device) for _ in range(total_batch_size)])
                for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                    samples[key] = samples[key][torch.arange(total_batch_size, device=device)[:, None], perms, ]

                samples_batched = {k: v.reshape(-1, config.batch_size_train, *v.shape[1:]) for k, v in samples.items()}

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

                        advantages = torch.clamp(sample["advantages"], -config.adv_clip_max, config.adv_clip_max,)
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range, )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    output = reward_fn(points, model.latent, dists, config.constraint_type, constraint = constraint)[1]
                    solved_cost, solved_tour, basic_cost, penalty_count = output['solved_cost'], output['solved_tour'], output['basic_cost'], output['penalty_count']
                    gap = 100*(solved_cost-gt_cost) / gt_cost
                    if solved_cost<final_solved_cost:
                        best_epoch = epoch
                        best_inner_epoch = inner_epoch
                        final_solved_cost = solved_cost
                        final_gap = gap
                        final_basic_cost = basic_cost
                        final_penalty_count = penalty_count
                # tour_image = test_dataset.draw_tour(solved_tour, points)
                # save_figure(img = tour_image, path = f'./images/encoding/tour_idx{int(sample_idx)}_epoch{epoch}_inner{inner_epoch}.png')

        sample_idxes.append(int(sample_idx))
        solved_costs.append(final_solved_cost)
        init_costs.append(init_cost)
        gt_costs.append(gt_cost)
        final_gaps.append(final_gap)
        init_gaps.append(init_gap)
        epochs.append(best_epoch)
        inner_epochs.append(best_inner_epoch)
        basic_costs.append(final_basic_cost)
        penalty_counts.append(final_penalty_count)
        
        del loss
        gc.collect()
        torch.cuda.empty_cache()
        
        if sample_idx%config.save_freq==0:
            result_df = pd.DataFrame({
                'sample_idx' : sample_idxes,
                'best_epoch' : epochs,
                'best_inner_epoch' : inner_epochs,
                'solved_cost' : solved_costs,
                'init_cost' : init_costs,
                'gt_cost' : gt_costs,
                'basic_cost' : basic_costs,
                'penalty_count' : penalty_counts,
                'final_gap(%)' : final_gaps,
                'init_gap(%)' : init_gaps,
            })
            if config.save_result:
                if not os.path.exists(f'./Results/{config.constraint_type}/{config.run_name}'):
                    os.makedirs(f'./Results/{config.constraint_type}/{config.run_name}')
                result_df.to_csv(f'./Results/{config.constraint_type}/{config.run_name}/{config.result_file_name}', index=False)

    else:        
        result_df = pd.DataFrame({
            'sample_idx' : sample_idxes,
            'best_epoch' : epochs,
            'best_inner_epoch' : inner_epochs,
            'solved_cost' : solved_costs,
            'init_cost' : init_costs,
            'gt_cost' : gt_costs,
            'basic_cost' : basic_costs,
            'penalty_count' : penalty_counts,
            'final_gap(%)' : final_gaps,
            'init_gap(%)' : init_gaps,
        })
        if config.save_result:
            print('save result')
            if not os.path.exists(f'./Results/{config.constraint_type}/{config.run_name}'):
                os.makedirs(f'./Results/{config.constraint_type}/{config.run_name}')
            result_df.to_csv(f'./Results/{config.constraint_type}/{config.run_name}/{config.result_file_name}', index=False)
