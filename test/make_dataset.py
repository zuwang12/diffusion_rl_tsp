import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import cv2
from concorde.tsp import TSPSolver
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
plt.style.use("seaborn-v0_8-dark")
from utils import create_distance_matrix



class DeliveryEnvironment:
    def __init__(self, n_stops=100, method="traffic_box", **kwargs):
        self.n_stops = n_stops
        self.method = method
        self.stops = []
        self._generate_constraints(**kwargs)
        self._generate_stops()
        self._generate_q_values()

    def _generate_constraints(self, box_size=0.2, traffic_intensity=5):
        if self.method == "traffic_box":
            x_left = np.random.rand() * (1 - box_size)
            y_bottom = np.random.rand() * (1 - box_size)
            x_right = x_left + np.random.rand() * box_size
            y_top = y_bottom + np.random.rand() * box_size
            self.box = (x_left, x_right, y_bottom, y_top)
            self.traffic_intensity = traffic_intensity

    def _generate_stops(self):
        points = []
        while len(points) < self.n_stops:
            x, y = np.random.rand(2)
            if not self._is_in_box(x, y, self.box):
                points.append((x, y))
        xy = np.array(points)
        self.x = xy[:, 0]
        self.y = xy[:, 1]
        
        # Prepare points array
        self.points = np.column_stack((self.y, self.x))

    def _generate_q_values(self):
        xy = np.column_stack([self.x, self.y])
        self.q_stops = cdist(xy, xy)

    def _is_in_box(self, x, y, box):
        x_left, x_right, y_bottom, y_top = box
        return x >= x_left and x <= x_right and y >= y_bottom and y <= y_top

    def render(self, size=7):
        fig = plt.figure(figsize=(size, size))
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops")

        # Show stops
        ax.scatter(self.x, self.y, c="red", s=50)
        
        # Display coordinates of points
        for i, (xi, yi) in enumerate(zip(self.x, self.y)):
            ax.text(xi, yi, f'city{i} - ({xi:.2f}, {yi:.2f})', fontsize=6, ha='right')

        if hasattr(self, "box"):
            left, bottom = self.box[0], self.box[2]
            width = self.box[1] - self.box[0]
            height = self.box[3] - self.box[2]
            rect = Rectangle((left, bottom), width, height)
            collection = PatchCollection([rect], facecolor="red", alpha=0.2)
            ax.add_collection(collection)
            
            # Display coordinates of rectangle corners
            corners = [(left, bottom), 
                       (left + width, bottom), 
                       (left, bottom + height), 
                       (left + width, bottom + height)]
            for i, (cx, cy) in enumerate(corners):
                ax.text(cx, cy, f'({cx:.2f}, {cy:.2f})', fontsize=9, ha='center', fontweight='bold')

        plt.xticks([])
        plt.yticks([])
        plt.show()
        
    def get_img(self, tour=None, point_circle = True, img_size=64, line_color=0.5, line_thickness=1, point_radius=2, point_color=1, box_color=0.8):
        img = np.zeros((img_size, img_size))

        # Draw the box
        if hasattr(self, 'box'):
            x_left = int(self.box[0] * (img_size - 1))
            x_right = int(self.box[1] * (img_size - 1))
            y_bottom = int(self.box[2] * (img_size - 1))
            y_top = int(self.box[3] * (img_size - 1))
            img[y_bottom:y_top, x_left:x_right] = box_color

        # Rasterize lines
        if isinstance(tour, np.ndarray):
            for i in range(tour.shape[0] - 1):
                from_idx = tour[i] - 1
                to_idx = tour[i + 1] - 1

                cv2.line(img,
                         tuple(((img_size-1)*self.points[from_idx, ::-1]).astype(int)),
                         tuple(((img_size-1)*self.points[to_idx, ::-1]).astype(int)),
                         color=line_color, thickness=line_thickness)

        # Rasterize points
        for i in range(self.points.shape[0]):
            if point_circle:
                cv2.circle(img, tuple(((img_size-1)*self.points[i, ::-1]).astype(int)), 
                           radius=point_radius, color=point_color, thickness=-1)
            else:
                row = round((img_size-1)*self.points[i,0])
                col = round((img_size-1)*self.points[i,1])
                img[row,col] = point_color
        # Rescale image to [-1,1]
        img = 2 * (img - 0.5)
        
        # 이미지 상하 반전
        img_flipped = np.flipud(img)
        
        return img_flipped

        
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


# Step 2: Create the distance matrix with a penalty for crossing the restricted zone
def create_distance_matrix2(env):
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    city_coords = np.column_stack((env.x, env.y))
    restricted_zone = env.box
    num_cities = len(city_coords)
    distance_matrix = {}
    penalty = 1e2  # Very large penalty for crossing the restricted zone

    for from_node in range(num_cities):
        distance_matrix[from_node] = {}
        for to_node in range(num_cities):
            if from_node == to_node:
                distance_matrix[from_node][to_node] = 0
            else:
                p1, p2 = city_coords[from_node], city_coords[to_node]
                distance = euclidean_distance(p1, p2)
                if crosses_restricted_zone(p1, p2, restricted_zone):
                    distance += penalty
                distance_matrix[from_node][to_node] = distance
    
    return distance_matrix
    
def write_tsplib_file(distance_matrix, filename, scale_factor=1000):
    size = len(distance_matrix)
    with open(filename, 'w') as f:
        f.write("NAME: TSP\n")
        f.write("TYPE: TSP\n")
        f.write("DIMENSION: {}\n".format(size))
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(size):
            for j in range(size):
                # Scale and convert float to int
                f.write("{} ".format(int(distance_matrix[i][j] * scale_factor)))
            f.write("\n")
        f.write("EOF\n")
    
# Plot the optimal route
def plot_route(route, env):
    plt.figure(figsize=(7, 7))
    plt.scatter(env.x, env.y, c='red', s=50)
    
    # Display coordinates of points
    for i, (xi, yi) in enumerate(zip(env.x, env.y)):
        plt.text(xi, yi, f'city{i} - ({xi:.2f}, {yi:.2f})', fontsize=6, ha='right')

    for i in range(len(route) - 1):
        plt.plot([env.x[route[i]], env.x[route[i + 1]]], [env.y[route[i]], env.y[route[i + 1]]], 'bo-')
    
    if hasattr(env, "box"):
        left, bottom = env.box[0], env.box[2]
        width = env.box[1] - env.box[0]
        height = env.box[3] - env.box[2]
        rect = Rectangle((left, bottom), width, height)
        collection = PatchCollection([rect], facecolor="red", alpha=0.2)
        plt.gca().add_collection(collection)
    
    plt.title("Optimal Delivery Route")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
def has_duplicates(l):
    existing = []
    for item in l:
        if item in existing:
            return True
        existing.append(item)
    return False


if __name__=='__main__':
    problem_cnt = 0
    target_problem_cnt = 1280

    with open('./data/tsp200_constraint_concorde_new.txt', 'w') as f:
        # tqdm 객체 생성
        with tqdm(total=target_problem_cnt, desc="Generating TSP problems", unit="problem") as pbar:
            while problem_cnt < target_problem_cnt:
                try:
                    env = DeliveryEnvironment(n_stops=200, method="traffic_box", box_size=0.6, traffic_intensity=100)
                    distance_matrix = create_distance_matrix2(env)
                    write_tsplib_file(distance_matrix, './data/tsp_problem.tsp')
                    solver = TSPSolver.from_tspfile('./data/tsp_problem.tsp')
                    solution = solver.solve()
                    
                    if solution is None:
                        print("No solution found, skipping this instance.")
                        continue
                    
                    route = solution.tour
                    tour = np.append(route, route[0]) + 1
                    if has_duplicates(tour[:-1]):
                        continue

                    flag = 0
                    box_matrix = create_distance_matrix(env.points.reshape([1] + list(env.points.shape)), np.array(env.box).reshape(1, 4), type='hard')
                    for i in range(len(tour)-1):
                        if box_matrix[tour[i]-1][tour[i+1]-1]==1:
                            flag=1
                            break
                    if flag == 1:
                        continue
                    
                    str_points = str(env.points.flatten().tolist()).replace('[', '').replace(']', '').replace(',', '')
                    str_tour = str(tour.flatten().tolist()).replace('[', '').replace(']', '').replace(',', '')
                    str_box = str(env.box).replace('(', '').replace(')', '').replace(',', '')
                    f.writelines(f'{str_points} output {str_tour} output {str_box} \n')

                    problem_cnt += 1
                    pbar.update(1)  # tqdm 객체 업데이트
                    
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue