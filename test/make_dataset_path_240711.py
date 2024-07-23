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

def sampling_edge(gt_tour, points, sample_cnt=1):
    # Get the length of gt_tour
    n = len(gt_tour)
    edges = []
    while len(edges)<sample_cnt:
        while True:
            # Randomly select two different indices
            idx1, idx2 = random.sample(range(n), 2)
            
            # Ensure the selected indices are not consecutive
            if abs(idx1 - idx2) != 1:
                break
        
        num1 = gt_tour[idx1] - 1
        num2 = gt_tour[idx2] - 1
        new_edge = [min(num1, num2), max(num1, num2)]

        if new_edge not in edges and not check_edge_intersection(new_edge, edges, points):
            edges.append(new_edge)
    
    return edges

def calculate_distance_matrix(points, edges = []):
    """
    Calculate the distance matrix for the given points with a penalty for lines that intersect the optimal box.
    """
    num_points = points.shape[0]
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(points[i] - points[j])
            if [i, j] not in edges:
                distance += 100
            distance_matrix[i, j] = distance_matrix[j, i] = distance
    return distance_matrix

def write_tsplib_file(distance_matrix, filename, scale_factor=1000):
    """
    Write the distance matrix to a TSPLIB format file.
    """
    size = len(distance_matrix)
    with open(filename, 'w') as f:
        f.write("NAME: TSP\nTYPE: TSP\nDIMENSION: {}\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n".format(size))
        for row in distance_matrix:
            f.write(" ".join(str(int(val * scale_factor)) for val in row) + "\n")
        f.write("EOF\n")

# Helper function to determine the orientation of the ordered triplet (p, q, r)
def orientation(p, q, r):
    """
    Determine the orientation of the triplet (p, q, r).
    0 -> p, q and r are collinear
    1 -> Clockwise
    2 -> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise

# Helper function to check if point q lies on segment pr
def on_segment(p, q, r):
    """
    Check if point q lies on segment pr.
    """
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

# Helper function to check if two line segments (p1q1 and p2q2) intersect
def do_intersect(p1, q1, p2, q2):
    """
    Check if line segments (p1q1) and (p2q2) intersect.
    """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    if o1 != o2 and o3 != o4:
        return True
    
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    
    return False

def check_edge_intersection(new_edge, existing_edges, points):
    p1, q1 = points[new_edge[0]], points[new_edge[1]]
    for edge in existing_edges:
        p2, q2 = points[edge[0]], points[edge[1]]
        if do_intersect(p1, q1, p2, q2):
            return True
    return False

def check_tour_intersections(tour, points):
    """
    Check if there are any intersections between the line segments in the given tour.
    Ignore intersections between consecutive segments.
    """
    for i in range(len(tour) - 1):
        p1 = points[tour[i] - 1]
        q1 = points[tour[i + 1] - 1]
        for j in range(i + 2, len(tour) - 1):
            if i == 0 and j == len(tour) - 2:
                continue  # Skip intersection check between the first and last segments
            p2 = points[tour[j] - 1]
            q2 = points[tour[j + 1] - 1]
            if do_intersect(p1, q1, p2, q2):
                return True
    return False

def save_tour_image(tour, points, edges, file_path, dataset):
    img = dataset.draw_tour(tour, points, edges=edges)
    plt.imshow(img, cmap='viridis')
    plt.savefig(file_path)
    plt.close()
    
    
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Solve TSP with path constraint.')
    parser.add_argument('--num_cities', default=100, type=int, help='Number of cities in the TSP instance')
    parser.add_argument('--save_image', default=False, type=bool)
    parser.add_argument('--img_size', default=64, type=int)
    args = parser.parse_args()
    
    # Set constants for drawing and loading the dataset
    IMG_SIZE = args.img_size
    POINT_RADIUS = 2
    POINT_COLOR = 1
    POINT_CIRCLE = True
    LINE_THICKNESS = 2
    LINE_COLOR = 0.5
    FILE_NAME = f'tsp{args.num_cities}_test_concorde.txt'
    SAVE_IMAGE = args.save_image
    BATCH_SIZE_SAMPLE = 1

    # Create an instance of the TSPDataset
    test_dataset = TSPDataset(
        data_file=f'./{FILE_NAME}', 
        img_size=IMG_SIZE, 
        point_radius=POINT_RADIUS, 
        point_color=POINT_COLOR,
        point_circle=POINT_CIRCLE, 
        line_thickness=LINE_THICKNESS, 
        line_color=LINE_COLOR, 
        return_box=False, 
        show_position=False
    )

    # Iterate through the dataset and solve TSP for the selected instance
    with open(f'./data/tsp{args.num_cities}_path_constraint_test.txt', 'w') as f:
        for i in tqdm(range(len(test_dataset))):
        # for i in range(10):
            # Get points and ground truth tour from the dataset
            img, points, gt_tour, sample_idx = test_dataset[i]
            sample_cnt = i%4+1
            # Sample initial edges and solve the TSP
            # edges = sampling_edge(gt_tour, points, sample_cnt=sample_cnt)
            # distance_matrix = calculate_distance_matrix(points, edges)
            # write_tsplib_file(distance_matrix, './test/tsp_problem.tsp')
            # with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            #     solver = TSPSolver.from_tspfile('./test/tsp_problem.tsp')
            #     solution = solver.solve()
            # route = np.append(solution.tour, solution.tour[0]) + 1
            
            # Continue sampling edges until a valid tour is found
            while True:
                edges = sampling_edge(gt_tour, points, sample_cnt=sample_cnt)
                distance_matrix = calculate_distance_matrix(points, edges)
                write_tsplib_file(distance_matrix, './test/tsp_problem.tsp')
                with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                    solver = TSPSolver.from_tspfile('./test/tsp_problem.tsp')
                    solution = solver.solve()
                route = np.append(solution.tour, solution.tour[0]) + 1

                if not check_tour_intersections(route, points):
                    if SAVE_IMAGE:
                        save_tour_image(gt_tour, points, edges, f'./images/path_constraint_{args.num_cities}/{i}_final_image.png', test_dataset)
                        save_tour_image(route, points, None, f'./images/path_constraint_{args.num_cities}/{i}_final_solved_image.png', test_dataset)
                    break
            
            flattened_edges = [str(node) for edge in edges for node in edge]
            path_constraint = ' '.join(flattened_edges)
                
            # Write results to file
            str_points = str(points.flatten().tolist()).replace('[', '').replace(']', '').replace(',', '')
            str_tour = str(route.flatten().tolist()).replace('[', '').replace(']', '').replace(',', '')
            str_path = path_constraint
            f.writelines(f'{str_points} output {str_tour} output {str_path} \n')