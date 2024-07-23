import numpy as np
from concorde.tsp import TSPSolver
from model.TSPModel import TSPDataset
from utils import adjust_distances_for_clusters, calculate_distance_matrix, write_tsplib_file
from tqdm import tqdm
import argparse
import os
from glob import glob
import sys
from sklearn.cluster import KMeans
import torch

def main():
    # 특정 경로 추가
    path_to_add = '/mnt/home/zuwang/workspace/diffusion_rl_tsp'
    if path_to_add not in sys.path:
        sys.path.append(path_to_add)

    # 특정 경로 제거
    path_to_remove = '/mnt/home/zuwang/workspace/ddpo-pytorch'
    if path_to_remove in sys.path:
        sys.path.remove(path_to_remove)
        
    parser = argparse.ArgumentParser(description='Solve TSP with box constraints.')
    parser.add_argument('--num_cities', default=20, help='Number of cities in the TSP instance')
    parser.add_argument('--save_image', default=False, type=bool, help='Save image or not')
    parser.add_argument('--img_size', default=64, type=int, help='Image size')
    args = parser.parse_args()

    # Define path
    root_path = '/mnt/home/zuwang/workspace/diffusion_rl_tsp'
    data_path = os.path.join(root_path, 'data')
    txt_path = os.path.join(data_path, f'tsp{args.num_cities}_cluster_constraint_test.txt')
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
        data_file=os.path.join(data_path, file_name), 
        img_size=img_size, 
        point_radius=point_radius, 
        point_color=point_color,
        point_circle=point_circle, 
        line_thickness=line_thickness, 
        line_color=line_color, 
        return_box=False, 
        show_position=False
    )

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    with open(txt_path, 'w') as f:
        for img, points, gt_tour, sample_idx in tqdm(test_dataloader):
            k = int(sample_idx[0]) % 4 + 3  # 3,4,5,6,3,4,5,6,...
            kmeans = KMeans(n_clusters=k, random_state=0).fit(points[0])
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            
            distance_matrix = calculate_distance_matrix(points[0])
            adjusted_distance_matrix = adjust_distances_for_clusters(distance_matrix, labels)
            
            write_tsplib_file(adjusted_distance_matrix, problem_path)
            solver = TSPSolver.from_tspfile(problem_path)
            solution = solver.solve()
            if solution:
                route = np.append(solution.tour, solution.tour[0]) + 1
                
            str_points = str(points[0].flatten().tolist()).replace('[', '').replace(']', '').replace(',', '')
            str_tour = str(route.flatten().tolist()).replace('[', '').replace(']', '').replace(',', '')
            str_label = str(labels.flatten().tolist()).replace('[', '').replace(']', '').replace(',', '')
            f.writelines(f'{str_points} output {str_tour} output {str_label} \n')
            # f.flush()

            # Remove .res and .sol files
            for file_path in glob('*.res') + glob('*.sol') + glob(data_path + '/*.sol') + glob(data_path + '/*.res'):
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error deleting file {file_path}: {e}")

if __name__ == '__main__':
    main()