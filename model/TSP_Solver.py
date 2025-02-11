import scipy
from utils import check_consecutive_pair, do_lines_intersect, check_cluster_degree_violations

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