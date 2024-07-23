import json
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
import os
from copy import deepcopy

from model.unet import UNetModel
from model.TSPModel import Model_x0, TSPDataset
from utils import TSP_2opt, runlat, create_distance_matrix, calculate_distance_matrix, calculate_distance_matrix2, draw_tour_box

import reward_fns
from pipeline_with_logprob import pipeline_with_logprob
from ddim_with_logprob import ddim_step_with_logprob
from diffusers import StableDiffusionPipeline, DDIMScheduler

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="tsp", help="which training config to use")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID for this process")
    parser.add_argument("--num_cities", type=int, default=20, help="Number of cities for the TSP problem")
    args = parser.parse_args()
    # mapping from config name to config path
    config_mapping = {"tsp":  "./configs/train_configs.yaml"}
    with open(config_mapping[args.config_name]) as file:
        config_dict= yaml.safe_load(file)
        config = munchify(config_dict)
        config.gpu_id = args.gpu_id
        config.num_cities = args.num_cities
    return config   

if __name__=='__main__':
    config = load_config()
    tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
    now = time.strftime('%y%m%d_%H%M%S')
    # config.run_name += f'_{now}'
    config.run_name = f'{now}_constraint_{config.num_cities}cities_gpu{config.gpu_id}'
    if not os.path.exists(f'./Results/{config.run_name}'):
        os.makedirs(f'./Results/{config.run_name}')
    # device = ('cuda' if torch.cuda.is_available() else 'cpu')
    device = f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    config.file_name = f'tsp{config.num_cities}_box_constraint_240710.txt'
    config.result_file_name = f'ours_tsp{config.num_cities}_constraint_epoch{config.num_epochs}_inner{config.num_inner_epochs}_{config.run_name}.csv'
    print(json.dumps(config, indent=4))
    
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
    unet.load_state_dict(torch.load(f'./ckpt/unet50_64_8.pth', map_location="cuda"))
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
        
    test_dataset = TSPDataset(data_file=f'./data/{config.file_name}',
                              img_size = config.img_size,
                              return_box = True,
                              point_radius = config.point_radius,
                              point_color = config.point_color,
                              point_circle = config.point_circle,
                              line_thickness = config.line_thickness,
                              line_color = config.line_color,
                              show_position=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size_sample, shuffle=False)
    num_points = test_dataset.rasterize(0)[1].shape[0]
    print('Created dataset')
    
    sample_idxes, solved_costs, init_costs, gt_costs, final_gaps, init_gaps, epochs, inner_epochs = [], [], [], [], [], [], [], []

    for img, points, gt_tour, sample_idx, box in tqdm(test_dataloader):
        # if int(sample_idx) not in [2]:continue
        ########### prepare constraint ##############
        # distance_matrix = create_distance_matrix(points, box, type='soft')
        # box_matrix = create_distance_matrix(points, box, type='hard')
        distance_matrix, intersection_matrix = calculate_distance_matrix2(points[0], box[0])
        
        ########### add prior model & prepare image ###########
        xT = torch.randn_like(img).float().to(device) # used at each problem
        model = Model_x0( # TODO: From define -> To reinit
            batch_size = config.batch_size_sample,
            num_points = num_points,
            img_size = config.img_size,
            line_color = config.line_color,
            line_thickness = config.line_thickness,
            xT = xT).to(device) #TODO: check batch_size from sample vs train
        model.eval()

        # _, points, gt_tour = test_dataset.rasterize(sample_idx[0].item())
        points, gt_tour, box = points.numpy()[0], gt_tour.numpy()[0], box.numpy()[0]
        solver = TSP_2opt(points)
        gt_cost = solver.evaluate([i-1 for i in gt_tour])
        img_query = torch.zeros_like(img)
        img_query[img == 1] = 1
        model.compute_edge_images(points=points, img_query=img_query) # Pre-compute edge images
        
        dists = np.zeros_like(model.latent[0].cpu().detach()) # (50, 50)
        for i in range(dists.shape[0]):
            for j in range(dists.shape[0]):
                dists[i,j] = np.linalg.norm(points[i]-points[j])

        ########### add prior model & prepare image ###########
        gt_img = test_dataset.draw_tour(tour = gt_tour, points = points, box=box)
        # save_image(torch.tensor((gt_img+1)/2), f'./images/gt_img_sample{int(sample_idx)}_cost{round(float(gt_cost), 2)}.png')
        
        
        reward_fn = getattr(reward_fns, config.reward_type)()
        final_solved_cost = 10**10
        final_gap = 0
        
        # Train!
        for epoch in range(config.num_epochs):
            #################### SAMPLING ####################                
            if epoch == 0:
                num_sample = config.num_init_sample
            else:
                num_sample = 1
            samples = []
            best_reward = -10**10
            for i in range(num_sample):
                if epoch == 0:
                    model.reset() # get various x0 | TODO: model define vs weight init compare!
                    if config.use_plug_and_play:
                        runlat(model, unet, STEPS=config.num_steps, batch_size=1, device=device)   
                _, _, latents, log_probs = pipeline_with_logprob( # image, has_nsfw_concept, all_latents, all_log_probs
                    pipeline,
                    num_inference_steps = config.num_steps,
                    eta=config.eta,
                    output_type="latent", # latent : resolution = image resolution
                    model = model,
                    device = device
                )

                latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
                log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
                timesteps = pipeline.scheduler.timesteps.repeat(config.batch_size_sample, 1)  # (batch_size, num_steps)

                rewards = torch.as_tensor(reward_fn(points, model.latent, dists, intersection_matrix)[0], device=device)
                if config.use_best_sample & (float(rewards)>best_reward):
                    best_reward = float(rewards)
                    model_init = deepcopy(model.state_dict())

                samples.append(
                    {
                        "timesteps": timesteps,
                        "latents": latents[:, :-1],  # each entry is the latent before timestep t
                        "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                        "log_probs": log_probs,
                        "rewards": rewards,
                    }
                ) 
            if epoch==0:
                if config.use_best_sample:
                    model.load_state_dict(model_init)
                else:
                    model.reset()

            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
            
            # gather rewards across processes
            rewards = (samples["rewards"]).cpu().numpy()
            
            solved_cost = -best_reward
            gap = 100*(solved_cost-gt_cost) / gt_cost
            
            if epoch == 0: # default setting
                init_cost = solved_cost
                init_gap = gap
            ################################## Update records ##################################
            
            if len(rewards)>1:
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                advantages = rewards

            # ungather advantages; we only need to keep the entries corresponding to the samples on this process
            samples["advantages"] = (torch.as_tensor(advantages).reshape(config.num_processes, -1)[config.process_index].to(device))
            del samples["rewards"]

            total_batch_size, num_timesteps = samples["timesteps"].shape
            
            # num_inner_epochs = total_batch_size
            if epoch == 0:
                num_inner_epochs = config.num_inner_epochs
            else:
                num_inner_epochs = 1
            
            optimizer_cls = torch.optim.AdamW
            optimizer = optimizer_cls(
                model.parameters(), # unet.parameters()
                lr =config.learning_rate,
                betas = (config.adam_beta1, config.adam_beta2),
                weight_decay = config.adam_weight_decay,
                eps = config.adam_epsilon
            )    
            
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=num_inner_epochs)
            
            ######################################## TRAINING ########################################
            for inner_epoch in range(num_inner_epochs):
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

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        
                    ############# save best result ##############
                    output = reward_fn(points, model.latent, dists, intersection_matrix)[1]
                    solved_cost, solved_tour = output['solved_cost'], output['solved_tour']
                    # print('solved tour : ', solved_tour)
                    gap = 100*(solved_cost-gt_cost) / gt_cost
                    # print(f'sample index : {int(sample_idx)}, {i}.{inner_epoch}-iter, solved cost : {solved_cost}, gt cost : {gt_cost}')  
                    if solved_cost<final_solved_cost:
                        best_epoch = epoch
                        best_inner_epoch = inner_epoch
                        final_solved_cost = solved_cost
                        final_gap = gap
                    ############# save best result ##############
                
                # if epoch == 0:   
                #     scheduler.step()
            # model.save_image(f'./images/encode_img_sample{int(sample_idx)}_epoch{int(epoch)}.png')
            # solved_img = test_dataset.draw_tour(tour=solved_tour, points = points, box = box)
            # output_test = reward_fn(points, model.latent, dists, box_matrix)[1]
            # save_image(torch.tensor((solved_img+1)/2), f'./images/solved_image_sample{int(sample_idx)}_cost{round(solved_cost, 2)}_epoch{int(epoch)}.png')
        ################################## append Result at each sample index ##################################
        sample_idxes.append(int(sample_idx))
        solved_costs.append(final_solved_cost)
        init_costs.append(init_cost)
        gt_costs.append(gt_cost)
        final_gaps.append(final_gap)
        init_gaps.append(init_gap)
        epochs.append(best_epoch)
        inner_epochs.append(best_inner_epoch)
        
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
                'final_gap(%)' : final_gaps,
                'init_gap(%)' : init_gaps,
            })
            if config.save_result:
                # if not os.path.exists(f'./Results/{config.run_name}'):
                #     os.makedirs(f'./Results/{config.run_name}')
                result_df.to_csv(f'./Results/{config.run_name}/{config.result_file_name}', index=False)

    else:        
        result_df = pd.DataFrame({
            'sample_idx' : sample_idxes,
            'best_epoch' : epochs,
            'best_inner_epoch' : inner_epochs,
            'solved_cost' : solved_costs,
            'init_cost' : init_costs,
            'gt_cost' : gt_costs,
            'final_gap(%)' : final_gaps,
            'init_gap(%)' : init_gaps,
        })
        if config.save_result:
            print('save result')
            # if not os.path.exists(f'./Results/{config.run_name}'):
            #     os.makedirs(f'./Results/{config.run_name}')
            result_df.to_csv(f'./Results/{config.run_name}/{config.result_file_name}', index=False)