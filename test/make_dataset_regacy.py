import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import matplotlib.pyplot as plt
from concorde.tsp import TSPSolver
from model.TSPModel import TSPDataset
from tqdm import tqdm

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
    """
    Check if the box defined by (x_left, x_right, y_bottom, y_top) is valid.
    The box is valid if no points are inside it.
    """
    return all(not (x_left < px < x_right and y_bottom < py < y_top) for px, py in points)

# Function to calculate the intersection and overlap of the box with the ground truth tour
def calculate_intersection_and_overlap(x_left, x_right, y_bottom, y_top, points, gt_tour):
    """
    Calculate the number of intersections and the overlap area between the box and the ground truth tour.
    """
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
    """
    Find the optimal box coordinates that maximize the intersection and overlap with the given tour.
    """
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

# Function to calculate the distance matrix with penalties for intersections with the optimal box
def calculate_distance_matrix(points, optimal_box):
    """
    Calculate the distance matrix for the given points with a penalty for lines that intersect the optimal box.
    """
    num_points = points.shape[0]
    distance_matrix = np.zeros((num_points, num_points))
    x_left, x_right, y_bottom, y_top = optimal_box
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(points[i] - points[j])
            if does_intersect_box(points[i], points[j], optimal_box):
                distance += 100
            distance_matrix[i, j] = distance_matrix[j, i] = distance
    return distance_matrix

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
    """
    Check if the given tour intersects with the box.
    """
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

if __name__=='__main__':
    # Define constants for drawing and loading the dataset
    img_size = 1024
    point_radius = 2
    point_color = 1
    point_circle = True
    line_thickness = 2
    line_color = 0.5
    file_name = 'tsp20_test_concorde.txt'
    batch_size_sample = 1

    # Create an instance of the TSPDataset
    test_dataset = TSPDataset(data_file=f'./data/{file_name}', img_size=img_size, point_radius=point_radius, point_color=point_color,
                            point_circle=point_circle, line_thickness=line_thickness, line_color=line_color, return_box=False, show_position=True)

    # Iterate through the dataset and solve TSP for the selected instance
    with open('./data/tsp20_box_revision_filtered_240709.txt', 'w') as f:
        for i in tqdm(range(len(test_dataset))):
            
            # Get points and ground truth tour from the dataset
            img, points, gt_tour, sample_idx = test_dataset[i]

            # Find the optimal box coordinates
            optimal_box = find_optimal_box(points, gt_tour)

            # Draw the tour with the optimal box using the TSPDataset instance
            # image = test_dataset.draw_tour(gt_tour, points, optimal_box)

            # Calculate the distance matrix with penalties
            distance_matrix = calculate_distance_matrix(points, optimal_box)

            # Write the distance matrix to a TSPLIB format file
            write_tsplib_file(distance_matrix, './data/tsp_problem.tsp')

            # Solve the TSP problem using the Concorde TSP solver
            solver = TSPSolver.from_tspfile('./data/tsp_problem.tsp')
            solution = solver.solve()
            if solution is None:
                print("No solution found, skipping this instance.")
                continue

            # Get the route and draw the solved tour
            route = np.append(solution.tour, solution.tour[0]) + 1
            # solved_image = test_dataset.draw_tour(route, points, optimal_box)
            
            if check_tour_box_overlap(route, optimal_box, points) or \
            check_tour_intersections(route, points):
                continue
            
            str_points = str(points.flatten().tolist()).replace('[', '').replace(']', '').replace(',', '')
            str_tour = str(route.flatten().tolist()).replace('[', '').replace(']', '').replace(',', '')
            str_box = str(optimal_box).replace('(', '').replace(')', '').replace(',', '')
            f.writelines(f'{str_points} output {str_tour} output {str_box} \n')