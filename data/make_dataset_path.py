import numpy as np
import matplotlib.pyplot as plt
from concorde.tsp import TSPSolver
from model.TSPModel import TSPDataset
import random
import os
import contextlib
from tqdm import tqdm
import argparse
import sys
from glob import glob
from utils import sampling_edge, calculate_distance_matrix, write_tsplib_file, check_tour_intersections

def save_tour_image(tour, points, edges, input_path, dataset):
    img = dataset.draw_tour(tour, points, edges=edges)
    plt.imshow(img, cmap='viridis')
    plt.savefig(input_path)
    plt.close()
    
if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Solve TSP with path constraint.')
    parser.add_argument('--num_cities', default=20, type=int, help='Number of cities in the TSP instance')
    parser.add_argument('--save_image', default=False, type=bool)
    parser.add_argument('--img_size', default=64, type=int)
    args = parser.parse_args()
    
    # Define path
    root_path = './'
    data_path = os.path.join(root_path, 'data')
    input_path = f'./data/tsp{args.num_cities}_train_concorde.txt'
    output_path = os.path.join(data_path, f'tsp{args.num_cities}_path_constraint_for_train.txt')
    problem_path = os.path.join(root_path, 'test/tsp_problem.tsp')
    img_path = os.path.join(root_path, f'images/path_constraint_{args.num_cities}')
    
    # Set constants for drawing and loading the dataset
    IMG_SIZE = args.img_size
    POINT_RADIUS = 2
    POINT_COLOR = 1
    POINT_CIRCLE = True
    LINE_THICKNESS = 2
    LINE_COLOR = 0.5
    # FILE_NAME = f'tsp{args.num_cities}_test_concorde.txt'
    # input_path = os.path.join(data_path, FILE_NAME)
    SAVE_IMAGE = args.save_image
    BATCH_SIZE_SAMPLE = 1
    print('input path : ', input_path)
    # Create an instance of the TSPDataset
    test_dataset = TSPDataset(
        data_file=input_path, 
        img_size=IMG_SIZE, 
        point_radius=POINT_RADIUS, 
        point_color=POINT_COLOR,
        point_circle=POINT_CIRCLE, 
        line_thickness=LINE_THICKNESS, 
        line_color=LINE_COLOR, 
        # constraint_type='path', 
        show_position=False,
    )

    print(f'output path : {output_path}')
    print(f'args : {args}')
    
    # Iterate through the dataset and solve TSP for the selected instance
    with open(output_path, 'w') as f:
        for i in tqdm(range(len(test_dataset))):
        # for i in range(10):
            # Get points and ground truth tour from the dataset
            img, points, gt_tour, sample_idx, _ = test_dataset[i]
            sample_cnt = i%2+1
            if i==100:
                SAVE_IMAGE=False
            
            # Continue sampling edges until a valid tour is found
            while True:
                edges = sampling_edge(gt_tour, points, sample_cnt=sample_cnt)
                distance_matrix = calculate_distance_matrix(points, edges)
                write_tsplib_file(distance_matrix, problem_path)
                with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                    solver = TSPSolver.from_tspfile(problem_path)
                    solution = solver.solve()
                route = np.append(solution.tour, solution.tour[0]) + 1
                if SAVE_IMAGE:
                    save_tour_image(gt_tour, points, edges, os.path.join(img_path, f'{i}_final_image.png'), test_dataset)
                    save_tour_image(route, points, None, os.path.join(img_path, f'{i}_final_solved_image.png'), test_dataset)
                    
                if not check_tour_intersections(route, points):
                    break
                else:
                    continue
            
            flattened_edges = [str(node) for edge in edges for node in edge]
            path_constraint = ' '.join(flattened_edges)
                
            # Write results to file
            str_points = str(points.flatten().tolist()).replace('[', '').replace(']', '').replace(',', '')
            str_tour = str(route.flatten().tolist()).replace('[', '').replace(']', '').replace(',', '')
            str_path = path_constraint
            f.writelines(f'{str_points} output {str_tour} output {str_path} \n')
            f.flush() # only for debugging
            
            res_files = glob('*.res')
            sol_files = glob('*.sol')
            all_files = res_files + sol_files
            
            for file in all_files:
                try:
                    os.remove(file)
                    # print(f'Delted file: {file}')
                except OSError as e:
                    print(f'Error deleting file {file} : {e}')
            