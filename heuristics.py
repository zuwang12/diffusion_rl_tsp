import torch
from model.TSPModel import TSPDataset
from utils import TSP_2opt
from tqdm import tqdm
import pandas as pd

def main():
    # Set constants for drawing and loading the dataset
    NUM_CITIES = 20
    IMG_SIZE = 64
    POINT_RADIUS = 2
    POINT_COLOR = 1
    POINT_CIRCLE = True
    LINE_THICKNESS = 2
    LINE_COLOR = 0.5
    FILE_NAME = F'tsp{NUM_CITIES}_path_constraint_240711.txt'
    SAVE_IMAGE = False
    BATCH_SIZE_SAMPLE = 1

    # Create an instance of the TSPDataset
    test_dataset = TSPDataset(
        data_file=f'./data/{FILE_NAME}', 
        img_size=IMG_SIZE, 
        point_radius=POINT_RADIUS, 
        point_color=POINT_COLOR,
        point_circle=POINT_CIRCLE, 
        line_thickness=LINE_THICKNESS, 
        line_color=LINE_COLOR, 
        show_position=True,
        constraint_type='path',
    )

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    solved_costs, penalty_counts, sample_idxes = [], [], []
    costs = 0
    for img, points, gt_tour, sample_idx, constraint in tqdm(test_dataloader):
        # if int(sample_idx)>5:break
        # distance_matrix, intersection_matrix = calculate_distance_matrix2(points[0], constraint[0])
        tsp_solver = TSP_2opt(points[0], path = constraint[0])
        tour = list(range(len(gt_tour[0])-1))
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
    
    result = pd.DataFrame({'sample_idxes' : sample_idxes,
                           'penalty_counts' : penalty_counts,
                           'solved_costs' : solved_costs,})
    
    result.to_csv('./Results/heuristics.csv', encoding='cp949')
if __name__=='__main__':
    main()

