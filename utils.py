import scipy
import numpy as np
import math
import gc
import random
import torch
import torch.nn.functional as F
from model.diffusion import GaussianDiffusion


# Set seed for reproducibility
seed_value = 2024
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

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

def check_consecutive_pair(lst, a, b):
    for i in range(len(lst) - 1):
        if (lst[i] == a and lst[i + 1] == b) or (lst[i] == b and lst[i + 1] == a):
            return True
    return False

def runlat(model, unet, STEPS, batch_size, device, use_fp16=True):
    """
    Function to initialize parameters without using accelerator.
    """
    # Initialize optimizer and scheduler (no accelerator involved)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0, 0.9))
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor=0.1, total_iters=1000)
    diffusion = GaussianDiffusion(T=1000, schedule='linear')

    # Initialize GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    steps = STEPS
    for i in range(steps):
        t = ((steps - i) + (steps - i) // 3 * math.cos(i / 50)) / steps * diffusion.T
        t = np.clip(t, 1, diffusion.T)
        t = np.array([t for _ in range(batch_size)]).astype(int)

        # Encode and sample from the diffusion model
        xt, epsilon = diffusion.sample(model.encode(), t)
        t = torch.from_numpy(t).float().view(batch_size).to(device)

        # Use torch.cuda.amp's autocast for mixed precision
        with torch.cuda.amp.autocast():
            xt = xt.to(device, dtype=torch.float32)
            epsilon = epsilon.to(device, dtype=torch.float32)

            # Denoise step
            epsilon_pred = unet(xt, t, use_fp16=use_fp16)
            loss = F.mse_loss(epsilon_pred, epsilon)

        # Use scaler for mixed precision
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        scheduler.step()

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

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

############################### box constraint ###############################

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

# Function to check if adding a new edge will create an intersection : 현재까지 연결된 Edge(real_adj_mat) 기준으로 intersection (with (a, b)) 체크
def check_for_intersection(a, b, real_adj_mat, points):
    for i in range(real_adj_mat.shape[0]):
        for j in range(i + 1, real_adj_mat.shape[0]):
            if real_adj_mat[i, j] == 1:
                if do_intersect(points[a], points[b], points[i], points[j]):
                    return True
    return False

# tour 기준으로 intersection (with (a, b)) 체크
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

#################################### cluster constraint ####################################

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