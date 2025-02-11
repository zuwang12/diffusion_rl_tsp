import numpy as np
from utils import construct_tsp_from_mst, check_for_intersection, make_tours_greedy, make_tours, batched_two_opt_torch
from model.TSP_Solver import TSP_2opt

def tsp_constraint():
    def has_duplicates(l):
        existing = []
        for item in l:
            if item in existing:
                return True
            existing.append(item)
        return False
    
    def _fn(points, adj_mat, dists, constraint_type=None, constraint=None, max_iter=1000):
        if constraint_type == 'basic':
            # adj_mat = normalize(model_latent).detach().cpu().numpy()
            # tours, _ = make_tours(adj_mat=adj_mat, np_points=points, edge_index_np=None, sparse_graph=False, parallel_sampling=1)
            tours, _ = make_tours(adj_mat=adj_mat.detach().cpu().numpy(), np_points=points)
            solved_tours, _ = batched_two_opt_torch(
                points.astype("float64"), np.array(tours).astype('int64'),
                max_iterations=max_iter, device=adj_mat.device)
            solved_tour = solved_tours[0]
            tsp_solver = TSP_2opt(points, constraint_type, constraint)
            
        else:
            # adj_mat = normalize(model_latent).detach().cpu().numpy()[0]
            tours = make_tours_greedy(adj_mat=adj_mat.detach().cpu().numpy()[0], points=points, dists=dists, constraint_type=constraint_type, constraint=constraint)
            tsp_solver = TSP_2opt(points, constraint_type, constraint)
            solved_tour, _ = tsp_solver.solve_2opt(tours, max_iter)

        assert solved_tour[-1] == solved_tour[0], 'Tour not a cycle'
        # assert not has_duplicates(solved_tour[:-1]), 'Tour not Hamiltonian'

        solved_cost = tsp_solver.evaluate(solved_tour)
        penalty_count = tsp_solver.count_constraints(solved_tour)

        return -np.array([solved_cost]), {
            'solved_tour': np.array(solved_tour) + 1,
            'basic_cost': solved_cost,
            'penalty_count': penalty_count,
        }

    return _fn