import torch
from model.TSPModel import TSPDataset
from utils import TSP_2opt
from tqdm import tqdm
import pandas as pd
import time
import argparse
import os
from utils import calculate_distance_matrix2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cities", type=int, default=20)
    parser.add_argument("--constraint_type", type=str, default='path')
    parser.add_argument("--save_freq", type=int, default=2)
    parser.add_argument("--run_name", type=str, default='2opt_test')
    args = parser.parse_args()
    # Set constants for drawing and loading the dataset
    date_per_type = {
       'box' : '240710',
       'path' : '240711',
       'cluster' : '240721', 
    }
    
    IMG_SIZE = 64
    POINT_RADIUS = 2
    POINT_COLOR = 1
    POINT_CIRCLE = True
    LINE_THICKNESS = 2
    LINE_COLOR = 0.5
    FILE_NAME = F'tsp{args.num_cities}_{args.constraint_type}_constraint_{date_per_type[args.constraint_type]}.txt'
    SAVE_IMAGE = False
    BATCH_SIZE_SAMPLE = 1
    now = time.strftime('%y%m%d_%H%M%S')
    
    root_path = '/mnt/home/zuwang/workspace/diffusion_rl_tsp'
    data_path = os.path.join(root_path, 'data')
    input_path = os.path.join(data_path, FILE_NAME)
    output_path = os.path.join(root_path, f'Results/{args.constraint_type}/{args.run_name}.csv')
    
    # Create an instance of the TSPDataset
    test_dataset = TSPDataset(
        data_file=input_path, 
        img_size=IMG_SIZE, 
        point_radius=POINT_RADIUS, 
        point_color=POINT_COLOR,
        point_circle=POINT_CIRCLE, 
        line_thickness=LINE_THICKNESS, 
        line_color=LINE_COLOR, 
        show_position=False,
        constraint_type=args.constraint_type,
    )

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    solved_costs, penalty_counts, sample_idxes = [], [], []
    costs = 0
    for img, points, gt_tour, sample_idx, constraint in tqdm(test_dataloader):
        img, points, gt_tour, sample_idx, constraint = (tensor.squeeze(0) for tensor in (img, points, gt_tour, sample_idx, constraint))
        # if int(sample_idx)>5:break
        if args.constraint_type=='box':
            distance_matrix, intersection_matrix = calculate_distance_matrix2(points, constraint)
            constraint = intersection_matrix
        tsp_solver = TSP_2opt(points, constraint_type=args.constraint_type, constraint = constraint)
        tour = list(range(len(gt_tour)-1))
        tour.append(0)
        solved_tour, _ = tsp_solver.solve_2opt(tour)
        solved_cost = tsp_solver.evaluate(solved_tour)
        # Calculate the penalty for constraints
        penalty_const = 10  # Define a penalty constant
        penalty_count = tsp_solver.count_constraints(solved_tour)  # Count the number of constraint violations
        # penalty = penalty_count * penalty_const  # Calculate the penalty
        # total_cost = solved_cost + penalty
        # costs += total_cost
        solved_costs.append(solved_cost)
        penalty_counts.append(penalty_count)
        sample_idxes.append(int(sample_idx))
        if int(sample_idx)%args.save_freq == 0:
            result = pd.DataFrame({'sample_idxes' : sample_idxes,
                            'penalty_counts' : penalty_counts,
                                'solved_costs' : solved_costs,})
            result.to_csv(output_path, encoding='cp949')
    else:
        result = pd.DataFrame({'sample_idxes' : sample_idxes,
                           'penalty_counts' : penalty_counts,
                            'solved_costs' : solved_costs,})
        result.to_csv(output_path, encoding='cp949')

if __name__=='__main__':
    main()

