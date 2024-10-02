import scipy
import numpy as np
import math
import gc
import random
import torch
import torch.nn.functional as F
from model.diffusion import GaussianDiffusion
from scipy.spatial import ConvexHull
from matplotlib.path import Path


# Set seed for reproducibility
seed_value = 2024
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

import numpy as np
import matplotlib.pyplot as plt
import cv2

def display_img(img, points, constraint_type='basic', constraint=None, tour=None, save_path = './test.png'):
    # Clip the values between -1 and 0
    img = np.clip(img, -1, 0)

    # Initialize an empty RGB array with shape (64, 64, 3)
    if constraint_type == 'basic':
        rgb_image = np.ones((64, 64, 3))  # Set background to white for basic constraint
    else:
        # Use input image for other constraints with scaling for -1 to 0 range
        scaled_values = 1- (img + 1)  # Map -1 -> 0 (white), 0 -> 1 (black)
        rgb_image = np.stack([scaled_values, scaled_values, scaled_values], axis=-1)  # Grayscale

    point_radius = 2

    if constraint_type == 'basic' and tour is not None:
        # Connect the points in tour order using black lines
        for i in range(len(tour) - 1):
            from_idx = tour[i] - 1
            to_idx = tour[i + 1] - 1
            cv2.line(rgb_image,
                     tuple((points[from_idx][::-1] * (64 - 1)).astype(int)),
                     tuple((points[to_idx][::-1] * (64 - 1)).astype(int)),
                     color=[0, 0, 0], thickness=1)
        # Close the loop by connecting the last point to the first
        cv2.line(rgb_image,
                 tuple((points[tour[-1] - 1][::-1] * (64 - 1)).astype(int)),
                 tuple((points[tour[0] - 1][::-1] * (64 - 1)).astype(int)),
                 color=[0, 0, 0], thickness=1)
    
    
    # Draw points and constraints based on constraint_type
    for i in range(len(points)):
        # Reverse coordinates (swap x and y)
        point_coords = tuple(((points[i][::-1]) * (64 - 1)).astype(int))
        color = [0, 1, 0]  # Default color is green

        # If constraint_type is 'cluster', change the color based on the cluster
        if constraint_type == 'cluster' and constraint is not None:
            cluster_colors = {
                7: [0.9, 0.9, 0.9],    # Gray for cluster 0
                2: [0.1, 0.1, 0.9],    # Blue for cluster 1
                0: [0.9, 0.1, 0.1],    # Red for cluster 2
                1: [0.1, 0.9, 0.1],    # Green for cluster 3
                3: [0.7, 0.7, 0.2],    # Yellow for cluster 4
                4: [0.3, 0.3, 0.7],    # Purple for cluster 5
                5: [0.6, 0.6, 0.3],    # Brown for cluster 6
                6: [0.4, 0.4, 0.4],    # Dark gray for cluster 7
            }
            cluster_id = int(constraint[i])
            color = cluster_colors.get(cluster_id, [1, 1, 1])  # Use cluster-specific color if available

        # Draw the circle at the point's location
        cv2.circle(rgb_image, point_coords, radius=point_radius, color=color, thickness=-1)

    # Handle constraints
    if constraint_type == 'box' and constraint is not None:
        # Draw a red rectangle (box), adjusting coordinates for reversed x and y
        y_bottom = int(constraint[0] * (64 - 1))
        y_top = int(constraint[1] * (64 - 1))
        x_left = int(constraint[2] * (64 - 1))
        x_right = int(constraint[3] * (64 - 1))
        cv2.rectangle(rgb_image, (x_left, y_bottom), (x_right, y_top), color=[1, 0, 0], thickness=-1)

    elif constraint_type == 'path' and constraint is not None:
        # Draw red lines connecting specified path points
        for i in range(0, len(constraint), 2):
            from_idx = int(constraint[i])
            to_idx = int(constraint[i + 1])
            cv2.line(rgb_image,
                     tuple(((points[from_idx][::-1]) * (64 - 1)).astype(int)),
                     tuple(((points[to_idx][::-1]) * (64 - 1)).astype(int)),
                     color=[1, 0, 0], thickness=2)

    # Display the RGB image
    plt.imshow(rgb_image)
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0)

class TSP_2opt:
    def __init__(self, points, constraint_type, constraint = None):
        self.points = points
        self.dist_mat = scipy.spatial.distance_matrix(points, points)
        self.constraint_type = constraint_type
        if constraint_type == 'box':
            self.constraint_matrix = constraint
        elif constraint_type == 'path':
            self.path_pairs = []
            for i in range(0, len(constraint), 2):
                self.path_pairs.append((int(constraint[i]), int(constraint[i+1])))
            self.path = constraint
        elif constraint_type == 'cluster':
            self.cluster = constraint

    def evaluate(self, route):
        return sum(self.dist_mat[route[i], route[i + 1]] for i in range(len(route) - 1))

    def count_constraints(self, route):
        count = 0
        if self.constraint_type == 'box':
            for i in range(len(route) - 1):
                if self.constraint_matrix[route[i], route[i + 1]] == 1:
                    count += 1

        if self.constraint_type == 'path':
            for a, b in self.path_pairs:
                if not check_consecutive_pair(route, a, b):
                    count += 1
                segment1 = (self.points[a], self.points[b])
                for j in range(len(route) - 1):
                    if bool(set([a, b]) & set([route[j], route[j + 1]])):
                        continue
                    segment2 = (self.points[route[j]], self.points[route[j + 1]])
                    if do_lines_intersect(segment1[0], segment1[1], segment2[0], segment2[1]):
                        count += 1

        if self.constraint_type == 'cluster':
            violations = check_cluster_degree_violations(self.cluster, route)
            count += violations
        return count

    def is_valid_route(self, route):
        for i in range(len(route)-1):
            if self.constraint_matrix[route[i], route[i+1]] == 1:
                return False
        return True

    def solve_2opt(self, route, max_iter = None):
        assert route[0] == route[-1], 'Tour is not a cycle'

        best = route
        best_constraints_cnt = self.count_constraints(route)
        best_cost = self.evaluate(best)
        improved = True
        steps = 0
        while improved:
            steps += 1
            if max_iter is not None and steps == max_iter:
                break
            improved = False
            for i in range(1, len(route) - 2):
                if self.constraint_type == 'path' and route[i] in self.path:
                    continue
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue

                    # Check if the edge (i, j) or (j, i) is in path_pairs
                    if self.constraint_type == 'path' and route[j] in self.path:
                        continue

                    new_route = route[:]
                    new_route[i:j] = route[j - 1:i - 1:-1]
                    new_constraints_cnt = self.count_constraints(new_route)
                    new_cost = self.evaluate(new_route)

                    if (new_cost < best_cost) and (new_constraints_cnt <= best_constraints_cnt):
                        if self.constraint_type != 'box' or self.is_valid_route(new_route):
                            best_cost = new_cost
                            best = new_route
                            best_constraints_cnt = new_constraints_cnt
                            improved = True

            route = best
        return best, steps

def runlat(model, unet, STEPS, batch_size, device):
    opt = torch.optim.Adam(model.parameters(), lr=1, betas=(0, 0.9))
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor=0.1, total_iters=1000)
    diffusion = GaussianDiffusion(T=1000, schedule='linear')
    # model.latent.data=temp

    steps = STEPS
    for i in range(steps):
        t = ((steps-i) + (steps-i)//3*math.cos(i/50))/steps*diffusion.T # Linearly decreasing + cosine

        t = np.clip(t, 1, diffusion.T)
        t = np.array([t for _ in range(batch_size)]).astype(int)

        # Denoise
        xt, epsilon = diffusion.sample(model.encode(), t) # get x_{ti} in Algorithm1 - (3 ~ 4)
        t = torch.from_numpy(t).float().view(batch_size)
        epsilon_pred = unet(xt.float(), t.to(device))

        loss = F.mse_loss(epsilon_pred, epsilon)

        loss.backward()
        opt.step()
        opt.zero_grad()
        scheduler.step()
    
    gc.collect()
    torch.cuda.empty_cache()
    
def check_consecutive_pair(lst, a, b):
    for i in range(len(lst) - 1):
        if (lst[i] == a and lst[i + 1] == b) or (lst[i] == b and lst[i + 1] == a):
            return True
    return False

def do_lines_intersect(p1, p2, q1, q2):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

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

# Function to check if adding a new edge will create an intersection
def check_for_intersection(a, b, real_adj_mat, points):
    for i in range(real_adj_mat.shape[0]):
        for j in range(i + 1, real_adj_mat.shape[0]):
            if real_adj_mat[i, j] == 1:
                if do_intersect(points[a], points[b], points[i], points[j]):
                    return True
    return False

def would_create_intersection(tour, new_edge, points):
    a, b = new_edge
    for i in range(len(tour) - 1):
        c, d = tour[i], tour[i + 1]
        if do_intersect(points[a], points[b], points[c], points[d]):
            return True
    return False

def construct_tsp_from_mst(adj_mat, real_adj_mat, dists, points, constraint_type = None, constraint = None):
    if constraint_type == 'box':
        constraint_matrix = constraint
    elif constraint_type == 'path':
        path = constraint
        mandatory_paths = set()
        for i in range(0, len(path), 2):
            a, b = int(path[i]), int(path[i + 1])
            mandatory_paths.add((a, b))
            mandatory_paths.add((b, a))
    elif constraint_type == 'cluster':
        cluster = constraint
    
    num_nodes = real_adj_mat.shape[0]
    tour = [0]
    visited = set(tour)
    adj_over_dists = adj_mat / dists

    while len(tour) < num_nodes:
        current_node = tour[-1]
        neighbors = np.nonzero(real_adj_mat[current_node])[0]
        next_node = None

        if constraint_type == 'path':
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                if (current_node, neighbor) in mandatory_paths:
                    next_node = neighbor
                    break

        if next_node is None:
            for neighbor in neighbors:
                if neighbor not in visited:
                    if not would_create_intersection(tour, (current_node, neighbor), points):
                        next_node = neighbor
                        break

        if next_node is None:
            remaining_nodes = list(set(range(num_nodes)) - visited)
            sorted_remaining_nodes = sorted(
                remaining_nodes, 
                key=lambda node: adj_over_dists[current_node, node], 
                reverse=True
            )

            for node in sorted_remaining_nodes:
                if constraint_type!='box' or constraint_matrix[current_node, node] == 0:
                    next_node = node
                    break
            else:
                next_node = random.choice(remaining_nodes)

        visited.add(next_node)
        tour.append(next_node)

    tour.append(0)
    return tour

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

############################### box constraint ###############################
# Function to check if the tour intersects with the box
def check_tour_box_overlap(tour, box, points):
    for i in range(len(tour) - 1):
        p1 = points[tour[i] - 1]
        p2 = points[tour[i + 1] - 1]
        if does_intersect_box(p1, p2, box):
            return True
    return False


# Function to calculate the intersection and overlap of the box with the ground truth tour
def calculate_intersection_and_overlap(x_left, x_right, y_bottom, y_top, points, gt_tour):
    intersection, overlap = 0, 0
    box_edges = [
        ((x_left, y_bottom), (x_left, y_top)), 
        ((x_left, y_top), (x_right, y_top)),
        ((x_right, y_top), (x_right, y_bottom)), 
        ((x_right, y_bottom), (x_left, y_bottom))
    ]
    for i in range(len(gt_tour) - 1):
        p1, q1 = points[gt_tour[i] - 1], points[gt_tour[i + 1] - 1]
        if any(do_intersect(p1, q1, p3, q3) for p3, q3 in box_edges):
            intersection += 1
            overlap += (min(x_right, q1[0]) - max(x_left, p1[0])) * (min(y_top, q1[1]) - max(y_bottom, p1[1]))
    return intersection, overlap

# Function to check if a box is valid by ensuring no points are inside it
def is_valid_box(x_left, x_right, y_bottom, y_top, points):
    return all(not (x_left < px < x_right and y_bottom < py < y_top) for px, py in points)

def find_optimal_box(points, gt_tour):
    hull_path = Path(points[ConvexHull(points).vertices])
    MIN_DISTANCE, MIN_EDGE_DISTANCE = 0.05, 0.05
    best_coords, max_intersection, max_overlap, max_area = None, 0, 0, float('inf')
    x_start, x_end, y_start, y_end = points[:,0].min(), points[:,0].max(), points[:,1].min(), points[:,1].max()

    for y_bottom in np.arange(y_start + MIN_EDGE_DISTANCE, y_end - MIN_EDGE_DISTANCE, 0.1):
        for x_left in np.arange(x_start + MIN_EDGE_DISTANCE, x_end - MIN_EDGE_DISTANCE, 0.1):
            for y_top in np.arange(y_bottom + 0.1, y_end - MIN_EDGE_DISTANCE, 0.1):
                for x_right in np.arange(x_left + 0.1, x_end - MIN_EDGE_DISTANCE, 0.1):
                    box_points = np.array([[x_left, y_bottom], [x_left, y_top], [x_right, y_bottom], [x_right, y_top]])
                    if not np.all(hull_path.contains_points(box_points)) or y_top <= y_bottom or x_right <= x_left:
                        continue
                    if not is_valid_box(x_left, x_right, y_bottom, y_top, points):
                        continue
                    intersection, overlap = calculate_intersection_and_overlap(x_left, x_right, y_bottom, y_top, points, gt_tour)
                    area = (y_top - y_bottom) * (x_right - x_left)
                    if (intersection > max_intersection or 
                        (intersection == max_intersection and overlap > max_overlap) or 
                        (intersection == max_intersection and overlap == max_overlap and area < max_area)):
                        best_coords = (x_left, x_right, y_bottom, y_top)
                        max_intersection, max_overlap, max_area = intersection, overlap, area
    if best_coords:
        print(f"Best coordinates: top-left ({best_coords[0]}, {best_coords[2]}), bottom-right ({best_coords[1]}, {best_coords[3]})")
    else:
        print("No valid rectangle found")
    return best_coords

def generate_random_box(gt_tour, points, width_max = 0.2, height_max = 0.2):
    def is_valid_box(box, points):
        (x_left, x_right, y_bottom, y_top) = box
        for x, y in points:
            if x_left <= x <= x_right and y_bottom <= y <= y_top:
                return False
        return True

    for _ in range(1000):  # 최대 1000번 시도
        idx = random.choice(range(len(gt_tour)-1))
        p1, p2 = points[gt_tour[idx] - 1], points[gt_tour[idx+1] - 1]
        cx, cy = (p1 + p2) / 2
        
        width = random.uniform(0.05, width_max)
        height = random.uniform(0.05, height_max)
        x_left, y_bottom = cx - width / 2, cy - height / 2
        x_right, y_top = cx + width / 2, cy + height / 2
        
        box = (x_left, x_right, y_bottom, y_top)
        
        if is_valid_box(box, points):
            return box


# Function to check if a line segment intersects with a given box
def does_intersect_box(p1, p2, box):
    """
    Check if the line segment (p1, p2) intersects with the given box.
    """
    x_left, x_right, y_bottom, y_top = box
    edges = [
        ((x_left, y_bottom), (x_left, y_top)), 
        ((x_left, y_top), (x_right, y_top)),
        ((x_right, y_top), (x_right, y_bottom)), 
        ((x_right, y_bottom), (x_left, y_bottom))
    ]
    return any(do_intersect(p1, p2, p3, q3) for p3, q3 in edges)

def calculate_distance_matrix2(points, box):
    """ 1 : impossibile, 0 : possible
    return both continuous, discrete distance matrix
    used for train_constraint.py

    Args:
        points (_type_): _description_
        box (_type_): _description_

    Returns:
        _type_: _description_
    """
    num_points = points.shape[0]
    distance_matrix = np.zeros((num_points, num_points))
    intersection_matrix = np.zeros((num_points, num_points))  # New matrix to indicate intersection with the box
    x_left, x_right, y_bottom, y_top = box
    
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(points[i] - points[j])
            if does_intersect_box(points[i], points[j], box):
                distance += 100
                intersection_matrix[i, j] = intersection_matrix[j, i] = 1  # Mark as intersecting with the box
            distance_matrix[i, j] = distance_matrix[j, i] = distance
            
    return distance_matrix, intersection_matrix

############################### box constraint ###############################

############################### path constraint ###############################
def sampling_edge(gt_tour, points, sample_cnt=1):
    """ function to sample edge path. used at path constraint generator

    Args:
        gt_tour (_type_): _description_
        points (_type_): _description_
        sample_cnt (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # Get the length of gt_tour
    n = len(gt_tour)
    edges = []
    while len(edges)<sample_cnt:
        while True:
            # Randomly select two different indices
            idx1, idx2 = random.sample(range(n), 2)
            
            # Ensure the selected indices are not consecutive
            if (abs(idx1 - idx2) != 1) & (idx1 != idx2) : # added at 20240926
                break
        
        num1 = gt_tour[idx1] - 1
        num2 = gt_tour[idx2] - 1
        new_edge = [min(num1, num2), max(num1, num2)]

        if new_edge not in edges and not check_edge_intersection(new_edge, edges, points):
            edges.append(new_edge)
    
    return edges

def check_edge_intersection(new_edge, existing_edges, points):
    p1, q1 = points[new_edge[0]], points[new_edge[1]]
    for edge in existing_edges:
        p2, q2 = points[edge[0]], points[edge[1]]
        if do_intersect(p1, q1, p2, q2):
            return True
    return False

def calculate_distance_matrix(points, edges=None, box=None):
    """
    Calculate the distance matrix for the given points with optional penalties for edges and intersections.
    
    Parameters:
    - points: A numpy array of shape (num_points, 2) representing the coordinates of the points.
    - edges: An optional list of tuples representing edges that should not have penalties. Default is None.
    - box: A tuple (x_left, x_right, y_bottom, y_top) representing the coordinates of a rectangular box.
           If an edge intersects this box, a penalty is added. Default is None.
           
    Returns:
    - distance_matrix: A numpy array of shape (num_points, num_points) representing the distance matrix
                       with penalties applied for edges not in the provided list and intersections with the box.
    """
    
    num_points = points.shape[0]
    distance_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(i + 1, num_points):
            # Calculate the Euclidean distance between points i and j
            distance = np.linalg.norm(points[i] - points[j])
            
            # If box is provided, check if the edge intersects the box
            if box is not None:
                if does_intersect_box(points[i], points[j], box):
                    distance += 100  # Add penalty for intersecting the box
            
            # If edges is provided and the edge (i, j) is not in the provided list of edges, add a penalty
            if edges is not None and ([i, j] not in edges and [j, i] not in edges):
                distance += 100
            
            distance_matrix[i, j] = distance_matrix[j, i] = distance
    
    return distance_matrix

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
############################### path constraint ###############################

#################################### cluster constraint ####################################

def adjust_distances_for_clusters(distance_matrix, labels, penalty=100):
    adjusted_matrix = distance_matrix.copy()
    num_points = distance_matrix.shape[0]
    
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if labels[i] != labels[j]:
                adjusted_matrix[i, j] += penalty
                adjusted_matrix[j, i] += penalty
                
    return adjusted_matrix

def check_cluster_degree_violations(cluster, solved_tour):
    cluster = [int(x) for x in cluster]
    cluster_degrees = {i: {'in': 0, 'out': 0} for i in np.unique(cluster)}

    for i in range(len(solved_tour) - 1):
        current_city = solved_tour[i]
        next_city = solved_tour[i + 1]

        current_cluster = cluster[current_city]
        next_cluster = cluster[next_city]

        if current_cluster != next_cluster:
            cluster_degrees[current_cluster]['out'] += 1
            cluster_degrees[next_cluster]['in'] += 1
    first_city = solved_tour[0]
    last_city = solved_tour[-1]

    first_cluster = cluster[first_city]
    last_cluster = cluster[last_city]

    if last_cluster != first_cluster:
        cluster_degrees[last_cluster]['out'] += 1
        cluster_degrees[first_cluster]['in'] += 1

    violations = 0
    for cluster_id, degrees in cluster_degrees.items():
        if degrees['in'] != 1 or degrees['out'] != 1:
            violations += 1

    return violations

#################################### cluster constraint ####################################