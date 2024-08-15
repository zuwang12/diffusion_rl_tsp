import scipy
import numpy as np
import cv2
import math
from tqdm import tqdm
import gc
import random

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from model.diffusion import GaussianDiffusion

from scipy.spatial import ConvexHull
from matplotlib.path import Path
from PIL import Image

seed_value = 2024

# Set seed for reproducibility
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)


class TSP_2opt():
    def __init__(self, points, constraint_type, constraint = None):
        # constraint_matrix=None, path = None, cluster = None
        self.points = points
        self.dist_mat = scipy.spatial.distance_matrix(points, points)
        self.constraint_type = constraint_type
        # If no constraint_matrix is provided, create one with all zeros (all connections allowed)
        # if constraint_matrix is None:
        #     self.constraint_matrix = np.zeros((points.shape[0], points.shape[0]))
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
        """_summary_

        Args:
            route (_type_): 0, 1, ... len(route)-1

        Returns:
            _type_: _description_
        """
        total_cost = 0
        for i in range(len(route)-1):
            total_cost += self.dist_mat[route[i],route[i+1]]
        return total_cost

    def count_constraints(self, route):
        count = 0
        if self.constraint_type == 'box':
            for i in range(len(route) - 1):
                if self.constraint_matrix[route[i], route[i + 1]] == 1:
                    count += 1
                    
        # Check for mandatory path overlaps
        if self.constraint_type == 'path':
            for path_pair in self.path_pairs:
                a, b = path_pair
                if not check_consecutive_pair(route, a, b):
                    count += 1
                segment1 = (self.points[a], self.points[b])
                for j in range(len(route) - 1):
                    if bool(set([a, b]) & set([route[j], route[j+1]])):
                    # if len(set([a, b]).intersection(set([route[j], route[j+1]])))!=0:
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
        best_constraints = self.count_constraints(route)
        improved = True
        steps = 0
        while improved:
            steps+=1
            if max_iter != None:
                if steps == max_iter:
                    break
            improved = False
            for i in range(1, len(route)-2):
                ############## path #############
                if self.constraint_type=='path':
                    if route[i] in self.path:
                        continue
                ############## path #############
                for j in range(i+1, len(route)):
                    if j-i == 1: 
                        continue # changes nothing, skip then
                    
                    ############## path #############
                    # Check if the edge (i, j) or (j, i) is in path_pairs
                    if self.constraint_type=='path':
                        if route[j] in self.path:
                            continue
                    ############## path #############

                    new_route = route[:]
                    new_route[i:j] = route[j-1:i-1:-1] # this is the 2optSwap
                    new_constraints = self.count_constraints(new_route)

                    if (self.evaluate(new_route) < self.evaluate(best)) and (new_constraints <= best_constraints):
                        if self.constraint_type != 'box' or self.is_valid_route(new_route):
                            best = new_route
                            improved = True
                            # steps += 1
                            if new_constraints < best_constraints:
                                best_constraints = new_constraints
                    
            route = best
        return best, steps
    
    def seed_solver(self, routes):
        best = self.solve_2opt(routes[0])
        for i in range(1, len(routes)):
            result = self.solve_2opt(routes[i])
            if self.evaluate(result) < self.evaluate(best):
                best = result
        return best
    
    def make_consecutive(self, lst, a, b):
        try:
            # a와 b의 인덱스를 찾음
            idx_a = lst.index(a)
            idx_b = lst.index(b)
            
            # a와 b가 이미 연속된 경우
            if abs(idx_a - idx_b) == 1:
                return lst  # 이미 연속되어 있으므로 변경할 필요 없음

            # a와 b가 연속되지 않은 경우
            if idx_a < idx_b:
                # a가 b보다 앞에 있는 경우
                lst.pop(idx_b)  # b를 리스트에서 제거
                lst.insert(idx_a + 1, b)  # a 바로 뒤에 b를 삽입
            else:
                # b가 a보다 앞에 있는 경우
                lst.pop(idx_a)  # a를 리스트에서 제거
                lst.insert(idx_b + 1, a)  # b 바로 뒤에 a를 삽입

        except ValueError as e:
            print(f"Error: {e}. One of the elements is not in the list.")

        return lst
    
def check_consecutive_pair(lst, a, b):
    for i in range(len(lst) - 1):
        if (lst[i] == a and lst[i + 1] == b) or (lst[i] == b and lst[i + 1] == a):
            return True
    return False
    
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
    
    
def normalize(cost, entropy_reg=0.1, n_iters=20, eps=1e-6):
    # Cost matrix is exp(-lambda*C)
    cost_matrix = -entropy_reg * cost # 0.1 * [1, 50, 50] (latent)
        
    cost_matrix -= torch.eye(cost_matrix.shape[-1], device=cost_matrix.device)*100000 # COST = COST - 100000*I
    cost_matrix = cost_matrix - torch.logsumexp(cost_matrix, dim=-1, keepdim=True)
    assignment_mat = torch.exp(cost_matrix)
    
    return assignment_mat # [1, 50, 50] (adj_mat)

def get_tsp_cost(points, model_latent, gt_tour):
    batch_size = 1 # TODO: what the hell is this?
    model_latent = torch.randn(batch_size,points.shape[0],points.shape[0])
    adj_mat = normalize((model_latent)).detach().cpu().numpy()[0] # model_latent : [1, 50, 50] -> adj_mat : (50, 50)
    adj_mat = adj_mat+adj_mat.T

    dists = np.zeros_like(adj_mat) # (50, 50)
    for i in range(dists.shape[0]):
        for j in range(dists.shape[0]):
            dists[i,j] = np.linalg.norm(points[i]-points[j])

    components = np.zeros((adj_mat.shape[0],2)).astype(int) # (50, 2)
    components[:] = np.arange(adj_mat.shape[0])[...,None] # (50, 1) | [[1], [2], ... , [49]]
    real_adj_mat = np.zeros_like(adj_mat) # (50, 50) 
    np.seterr(divide='ignore', invalid='ignore') # TODO: need to set error option globally
    for edge in (-adj_mat/dists).flatten().argsort(): # [1715,  784, 1335, ..., 1326, 1224, 2499]) | 실제 거리(dists) 대비 adj_mat값이 가장 높은 순으로 iter
        a,b = edge//adj_mat.shape[0],edge%adj_mat.shape[0] # (34, 15)
        if not (a in components and b in components): continue
        ca = np.nonzero((components==a).sum(1))[0][0] # 34
        cb = np.nonzero((components==b).sum(1))[0][0] # 15
        if ca==cb: continue
        cca = sorted(components[ca],key=lambda x:x==a) # [34, 34]
        ccb = sorted(components[cb],key=lambda x:x==b) # [15, 15]
        newc = np.array([[cca[0],ccb[0]]]) # [34, 15]
        m,M = min(ca,cb),max(ca,cb) # (15, 34)
        real_adj_mat[a,b] = 1 # 연결됨
        components = np.concatenate([components[:m],components[m+1:M],components[M+1:],newc],0) # (49, 2)
        if len(components)==1: break
    real_adj_mat[components[0,1],components[0,0]] = 1 # 마지막 연결
    real_adj_mat += real_adj_mat.T # make symmetric matrix

    tour = [0]
    while len(tour)<adj_mat.shape[0]+1:
        n = np.nonzero(real_adj_mat[tour[-1]])[0]
        if len(tour)>1:
            n = n[n!=tour[-2]]
        tour.append(n.max())

    # Refine using 2-opt
    tsp_solver = TSP_2opt(points)
    solved_tour, ns = tsp_solver.solve_2opt(tour)

    def has_duplicates(l):
        existing = []
        for item in l:
            if item in existing:
                return True
            existing.append(item)
        return False

    assert solved_tour[-1] == solved_tour[0], 'Tour not a cycle'
    assert not has_duplicates(solved_tour[:-1]), 'Tour not Hamiltonian'

    gt_cost = tsp_solver.evaluate([i-1 for i in gt_tour])
    solved_cost = tsp_solver.evaluate(solved_tour)
    # print(f'Ground truth cost: {gt_cost:.3f}') #TODO: need to delete -> return cost
    # print(f'Predicted cost: {solved_cost:.3f} (Gap: {100*(solved_cost-gt_cost) / gt_cost:.4f}%)')
    # costs.append((solved_cost, ns))
    
    return -np.array([solved_cost]), {'solved_tour' : solved_tour, 
                                        'points' : points, 
                                        'gt_cost' : gt_cost,
                                        'solved_cost' : solved_cost,}    
    
def draw_tour(tour, points, img_size = 64, line_color = 0.5, line_thickness = 2, point_circle = True, point_radius = 2, point_color = 1):
    img = np.zeros((img_size, img_size))
    # Rasterize lines
    for i in range(tour.shape[0]-1):
        from_idx = int(tour[i]-1)
        to_idx = int(tour[i+1]-1)

        cv2.line(img, 
                    tuple(((img_size-1)*points[from_idx,::-1]).astype(int)), 
                    tuple(((img_size-1)*points[to_idx,::-1]).astype(int)), 
                    color=line_color, thickness=line_thickness)

    # Rasterize points
    for i in range(points.shape[0]):
        if point_circle:
            cv2.circle(img, tuple(((img_size-1)*points[i,::-1]).astype(int)), 
                        radius=point_radius, color=point_color, thickness=-1)
        else:
            row = round((img_size-1)*points[i,0])
            col = round((img_size-1)*points[i,1])
            img[row,col] = point_color
        
    # Rescale image to [-1,1]
    img = 2*(img-0.5)
    return img
    
def draw_tour_box(tour, points, box = None, img_size = 64, line_color = 0.5, line_thickness = 2, point_circle = True, point_radius = 2, point_color = 1, box_color = 0.75):
    img = np.zeros((img_size, img_size))
    # Rasterize lines
    for i in range(tour.shape[0]-1):
        from_idx = int(tour[i]-1)
        to_idx = int(tour[i+1]-1)

        cv2.line(img, 
                    tuple(((img_size-1)*points[from_idx,::-1]).astype(int)), 
                    tuple(((img_size-1)*points[to_idx,::-1]).astype(int)), 
                    color=line_color, thickness=line_thickness)

    # Rasterize points
    for i in range(points.shape[0]):
        if point_circle:
            if i==0:
                cv2.circle(img, tuple(((img_size-1)*points[i,::-1]).astype(int)), 
                        radius=point_radius, color=0.25, thickness=-1)
            else:
                cv2.circle(img, tuple(((img_size-1)*points[i,::-1]).astype(int)), 
                        radius=point_radius, color=point_color, thickness=-1)
        else:
            row = round((img_size-1)*points[i,0])
            col = round((img_size-1)*points[i,1])
            img[row,col] = point_color
    
    if box is not None:
        x_left = int(box[0] * (img_size - 1))
        x_right = int(box[1] * (img_size - 1))
        y_bottom = int(box[2] * (img_size - 1))
        y_top = int(box[3] * (img_size - 1))
        img[y_bottom:y_top, x_left:x_right] = box_color
        
    # Rescale image to [-1,1]
    # img = 2*(img-0.5)
    return img
    
def rasterize_tsp(points, tour, img_size, line_color, line_thickness, point_color, point_radius):
    # Rasterize lines
    img = np.zeros((img_size, img_size))
    for i in range(len(tour)-1):
        from_idx = tour[i]
        to_idx = tour[i+1]

        cv2.line(img, 
                 tuple(((img_size-1)*points[from_idx,::-1]).astype(int)), 
                 tuple(((img_size-1)*points[to_idx,::-1]).astype(int)), 
                 color=line_color, thickness=line_thickness)

    # Rasterize points
    for i in range(points.shape[0]):
        cv2.circle(img, tuple(((img_size-1)*points[i,::-1]).astype(int)), 
                   radius=point_radius, color=point_color, thickness=-1)

    return img
    
########################################## Ragacy ################################################

# Function to check if two line segments intersect
def do_lines_intersect(p1, p2, q1, q2):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

# Function to check if a line segment intersects a rectangle
def crosses_restricted_zone(p1, p2, restricted_zone):
    x1, x2, y1, y2 = restricted_zone
    # Ensure that x1, y1 is the bottom-left and x2, y2 is the top-right
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Define the corners of the rectangle
    corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    
    # Define the edges of the rectangle
    edges = [
        (corners[0], corners[1]),  # Left edge
        (corners[1], corners[3]),  # Top edge
        (corners[3], corners[2]),  # Right edge
        (corners[2], corners[0])   # Bottom edge
    ]
    
    # Check if the line segment intersects any of the rectangle's edges
    for edge in edges:
        if do_lines_intersect(p1, p2, edge[0], edge[1]):
            return True
    return False

def create_distance_matrix(points, box, type='soft'):
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # points = np.column_stack((env.x, env.y))
    points = points[0]
    box = box[0]
    restricted_zone = box
    num_cities = len(points)
    distance_matrix = {}
    penalty = 1e2  # Very large penalty for crossing the restricted zone

    for from_node in range(num_cities):
        distance_matrix[from_node] = {}
        for to_node in range(num_cities):
            if from_node == to_node:
                distance_matrix[from_node][to_node] = 0
            else:
                p1, p2 = points[from_node], points[to_node]
                if type=='soft':
                    distance = euclidean_distance(p1, p2)
                    if crosses_restricted_zone(p1, p2, restricted_zone):
                        distance += penalty
                    distance_matrix[from_node][to_node] = distance
                elif type=='hard':
                    distance = 0
                    if crosses_restricted_zone(p1, p2, restricted_zone):
                        distance = 1
                    distance_matrix[from_node][to_node] = distance
    
    return distance_matrix

# def write_tsplib_file(distance_matrix, filename, scale_factor=1000):
#     size = len(distance_matrix)
#     with open(filename, 'w') as f:
#         f.write("NAME: TSP\n")
#         f.write("TYPE: TSP\n")
#         f.write("DIMENSION: {}\n".format(size))
#         f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
#         f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
#         f.write("EDGE_WEIGHT_SECTION\n")
#         for i in range(size):
#             for j in range(size):
#                 # Scale and convert float to int
#                 f.write("{} ".format(int(distance_matrix[i][j] * scale_factor)))
#             f.write("\n")
#         f.write("EOF\n")
        
        
def get_cost(points, tour):
    """ tour의 length를 계산하는 함수

    Args:
        points (np.array): (N, 2) shape
        tour (list): [0, 6, 9, 12, ... 0] length N+1의 list. 첫 시작과 마지막은 0으로 fix

    Returns:
        _type_: _description_
    """
    costs = []
    tsp_solver = TSP_2opt(points)
    cost = tsp_solver.evaluate(tour)
    costs.append(cost)
    
    return sum(costs)/len(costs)

###############################################################################################################


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

# Function to check if a box is valid by ensuring no points are inside it
def is_valid_box(x_left, x_right, y_bottom, y_top, points):
    return all(not (x_left < px < x_right and y_bottom < py < y_top) for px, py in points)

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

# Function to find the optimal box coordinates that maximize the intersection and overlap with the given tour
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

# Function to write the distance matrix to a TSPLIB format file
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

# Function to check if the tour intersects with the box
def check_tour_box_overlap(tour, box, points):
    for i in range(len(tour) - 1):
        p1 = points[tour[i] - 1]
        p2 = points[tour[i + 1] - 1]
        if does_intersect_box(p1, p2, box):
            return True
    return False

# Function to check if there are any intersections in the tour
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

# Function to check if a given tour is a valid TSP solution
def is_valid_tsp_solution(tour, num_nodes):
    return len(set(tour)) == num_nodes and len(tour) == num_nodes + 1 and tour[0] == tour[-1]

# TSP solution construction from real_adj_mat
def construct_tsp_from_mst(adj_mat, real_adj_mat, dists, points, constraint_type = None, constraint = None):
    if constraint_type == 'box':
        constraint_matrix = constraint
    elif constraint_type == 'path':
        path = constraint
    elif constraint_type == 'cluster':
        cluster = constraint
    
    num_nodes = real_adj_mat.shape[0]
    tour = [0]
    visited = set(tour)
    adj_over_dists = adj_mat / dists

    if constraint_type == 'path':
        mandatory_paths = set()
        for i in range(0, len(path), 2):
            a, b = int(path[i]), int(path[i + 1])
            mandatory_paths.add((a, b))
            mandatory_paths.add((b, a))

    while len(tour) < num_nodes:
        current_node = tour[-1]
        neighbors = np.nonzero(real_adj_mat[current_node])[0]
        next_node = None

        # Check for mandatory edges first
        if constraint_type == 'path':
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                if (current_node, neighbor) in mandatory_paths:
                    next_node = neighbor
                    break

        # If no mandatory edge found, proceed as usual
        if next_node is None:
            for neighbor in neighbors:
                if neighbor not in visited:
                    if not would_create_intersection(tour, (current_node, neighbor), points):
                        next_node = neighbor
                        break

        if next_node is None:  # No unvisited neighbor found
            # Select the node with the highest adj_over_dists value that has not been visited
            remaining_nodes = list(set(range(num_nodes)) - visited)
            sorted_remaining_nodes = sorted(remaining_nodes, key=lambda node: adj_over_dists[current_node, node], reverse=True)

            for node in sorted_remaining_nodes:
                if constraint_type!='box' or constraint_matrix[current_node, node] == 0:
                    next_node = node
                    break
            else:  # If no valid next node is found, randomly select from remaining nodes
                next_node = random.choice(remaining_nodes)

        visited.add(next_node)
        tour.append(next_node)

    tour.append(0)  # Return to start
    return tour


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
    # Initialize the dictionary to count the degree for each cluster
    cluster_degrees = {i: {'in': 0, 'out': 0} for i in np.unique(cluster)}

    # Check each edge in the solved_tour
    for i in range(len(solved_tour) - 1):
        current_city = solved_tour[i]
        next_city = solved_tour[i + 1]

        current_cluster = cluster[current_city]
        next_cluster = cluster[next_city]

        if current_cluster != next_cluster:
            # If moving to a different cluster, increment out-degree of current cluster
            # and in-degree of the next cluster
            cluster_degrees[current_cluster]['out'] += 1
            cluster_degrees[next_cluster]['in'] += 1

    # Check the last connection (to form a cycle)
    first_city = solved_tour[0]
    last_city = solved_tour[-1]

    first_cluster = cluster[first_city]
    last_cluster = cluster[last_city]

    if last_cluster != first_cluster:
        cluster_degrees[last_cluster]['out'] += 1
        cluster_degrees[first_cluster]['in'] += 1

    # Calculate the number of violations
    violations = 0
    for cluster_id, degrees in cluster_degrees.items():
        if degrees['in'] != 1 or degrees['out'] != 1:
            violations += 1
            # print(f"Cluster {cluster_id} has {degrees['in']} in-degrees and {degrees['out']} out-degrees.")

    return violations


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
            if abs(idx1 - idx2) != 1:
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

def save_figure(img, path):

    img -= img.min()
    img /= img.max()
    
    # Initialize an empty (64, 64, 3) array for the RGB image
    image_array = np.zeros((64, 64, 3), dtype=np.uint8)

    # Define the colors for interpolation
    white = np.array([255, 255, 255])   # Color for 0
    black = np.array([0, 0, 0])         # Color for 0.5
    green = np.array([0, 255, 0])       # Color for 1

    # Interpolate colors based on the value in `img`
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            value = img[i, j]
            if value <= 0.5:
                # Interpolate between white and black
                color = white * (0.5 - value) / 0.5 + black * value / 0.5
            else:
                # Interpolate between black and green
                color = black * (1.0 - value) / 0.5 + green * (value - 0.5) / 0.5
            
            image_array[i, j] = color

    # Convert the array back to an image
    image = Image.fromarray(image_array)

    # Save or display the image
    image.save(path)
    # image.show()