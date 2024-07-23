import numpy as np
from concorde.tsp import TSPSolver
from model.TSPModel import TSPDataset
from utils import find_optimal_box, calculate_distance_matrix, write_tsplib_file, check_tour_box_overlap, check_tour_intersections, generate_random_box
import random
import cv2
from tqdm import tqdm
import argparse
import os
from glob import glob
from matplotlib import pyplot as plt
import sys
from datetime import datetime
import time

def solve_tsp_instance(i, points, gt_tour, problem_path):
    optimal_box = find_optimal_box(points, gt_tour)
    if not optimal_box:
        return None

    distance_matrix = calculate_distance_matrix(points, optimal_box)
    write_tsplib_file(distance_matrix, problem_path)

    solver = TSPSolver.from_tspfile(problem_path)
    solution = solver.solve()
    if solution is None:
        return None

    route = np.append(solution.tour, solution.tour[0]) + 1
    if check_tour_box_overlap(route, optimal_box, points) or check_tour_intersections(route, points):
        return None

    return route, optimal_box

if __name__ == '__main__':
    start_time = time.time()
        # 특정 경로 추가
    path_to_add = '/mnt/home/zuwang/workspace/diffusion_rl_tsp'
    if path_to_add not in sys.path:
        sys.path.append(path_to_add)

    # 특정 경로 제거
    path_to_remove = '/mnt/home/zuwang/workspace/ddpo-pytorch'
    if path_to_remove in sys.path:
        sys.path.remove(path_to_remove)
        
    # Get today's date
    today_date = datetime.today().strftime('%y%m%d')
    
    parser = argparse.ArgumentParser(description='Solve TSP with box constraints.')
    parser.add_argument('--num_cities', default=20, help='Number of cities in the TSP instance')
    parser.add_argument('--save_image', default=False, type=bool, help='Save image or not')
    parser.add_argument('--img_size', default=64, type=int, help='Image size')
    args = parser.parse_args()

    # Define path
    root_path = '/mnt/home/zuwang/workspace/diffusion_rl_tsp'
    data_path = os.path.join(root_path, 'data')
    txt_path = os.path.join(data_path, f'tsp{args.num_cities}_box_constraint_test_{today_date}.txt')
    problem_path = os.path.join(root_path, 'problem/tsp_problem.tsp')
    img_path = os.path.join(root_path, f'images/box_constraint_{args.num_cities}')
    
    img_size = args.img_size
    point_radius = 2
    point_color = 1
    point_circle = True
    line_thickness = 2
    line_color = 0.5
    file_name = f'tsp{args.num_cities}_test_concorde.txt'
    batch_size_sample = 1
    save_image = args.save_image

    test_dataset = TSPDataset(
        data_file= os.path.join(data_path, file_name), 
        img_size=img_size, 
        point_radius=point_radius, 
        point_color=point_color,
        point_circle=point_circle, 
        line_thickness=line_thickness, 
        line_color=line_color, 
        constraint_type=None,
        show_position=False
    )



    with open(txt_path, 'w') as f:
        for i in tqdm(range(len(test_dataset))):
            img, points, gt_tour, sample_idx, _ = test_dataset[i]
            result = solve_tsp_instance(i, points, gt_tour, problem_path)
            while not result:
                random_box = generate_random_box(gt_tour, points)
                distance_matrix = calculate_distance_matrix(points, random_box)
                write_tsplib_file(distance_matrix, problem_path)
                solver = TSPSolver.from_tspfile(problem_path)
                solution = solver.solve()
                if solution:
                    route = np.append(solution.tour, solution.tour[0]) + 1
                    if not (check_tour_box_overlap(route, random_box, points) or check_tour_intersections(route, points)):
                        result = (route, random_box)

            route, optimal_box = result

            if save_image:
                first_image = test_dataset.draw_tour(gt_tour, points, optimal_box)
                plt.imshow(first_image, cmap='viridis')
                plt.savefig(img_path)
                plt.close()
                first_solved_image = test_dataset.draw_tour(route, points, optimal_box)
                plt.imshow(first_solved_image, cmap='viridis')
                plt.savefig(img_path)
                plt.close()

            str_points = str(points.flatten().tolist()).replace('[', '').replace(']', '').replace(',', '')
            str_tour = str(route.flatten().tolist()).replace('[', '').replace(']', '').replace(',', '')
            str_box = str(optimal_box).replace('(', '').replace(')', '').replace(',', '')
            f.writelines(f'{str_points} output {str_tour} output {str_box} \n')

            # Remove .res and .sol files
            for file_path in glob('*.res') + glob('*.sol') + glob(data_path + '/*.sol') + glob(data_path + '/*.res'):
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error deleting file {file_path}: {e}")
    
    print('total time : ', time.time() - start_time)