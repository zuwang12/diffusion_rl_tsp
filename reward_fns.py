import numpy as np
import torch
from utils import TSP_2opt, construct_tsp_from_mst, check_for_intersection

def tsp_constraint():
    def normalize(cost, entropy_reg=0.1):
        cost_matrix = -entropy_reg * cost
        cost_matrix -= torch.eye(cost_matrix.shape[-1], device=cost_matrix.device) * 100000
        cost_matrix -= torch.logsumexp(cost_matrix, dim=-1, keepdim=True)
        return torch.exp(cost_matrix)

    def _fn(points, model_latent, dists, constraint_type=None, constraint=None, max_iter=2):
        adj_mat = normalize(model_latent).detach().cpu().numpy()[0]
        adj_mat += adj_mat.T

        num_cities = adj_mat.shape[0]
        components = np.zeros((num_cities,2)).astype(int)
        components[:] = np.arange(num_cities)[...,None]
        real_adj_mat = np.zeros_like(adj_mat)

        np.seterr(divide='ignore', invalid='ignore')

        if constraint_type == 'box':
            constraint_matrix = constraint

        if constraint_type == 'path':
            path = constraint
            for i in range(0, len(path), 2):
                a, b = int(path[i]), int(path[i + 1])
                real_adj_mat[a, b] = 1
                ca = np.nonzero((components == a).sum(1))[0][0]
                cb = np.nonzero((components == b).sum(1))[0][0]
                cca = sorted(components[ca], key=lambda x: x == a)
                ccb = sorted(components[cb], key=lambda x: x == b)
                newc = np.array([[cca[0], ccb[0]]])
                m, M = min(ca, cb), max(ca, cb)
                components = np.concatenate([components[:m], components[m + 1:M], components[M + 1:], newc], 0)

        if constraint_type == 'cluster':
            cluster = constraint
            selected_nodes = set()  # Track selected clusters

        for edge in (-adj_mat/dists).flatten().argsort():
            a, b = edge//num_cities, edge%num_cities
            if a == b or (not (a in components and b in components)) or check_for_intersection(a, b, real_adj_mat, points):
                continue
            
            if constraint_type == 'box':
                if constraint_matrix[a][b] == 1:
                    continue

            # Ensure only one node per cluster is selected
            if constraint_type == 'cluster':
                if cluster[a] in selected_nodes and cluster[b] in selected_nodes:
                    continue

            ca = np.nonzero((components==a).sum(1))[0][0]
            cb = np.nonzero((components==b).sum(1))[0][0]
            if ca == cb:
                continue
            cca = sorted(components[ca],key=lambda x:x==a)
            ccb = sorted(components[cb],key=lambda x:x==b)
            newc = np.array([[cca[0],ccb[0]]]) # [34, 15]
            m,M = min(ca,cb),max(ca,cb) # (15, 34)
            real_adj_mat[a,b] = 1 # 연결됨
            components = np.concatenate([components[:m],components[m+1:M],components[M+1:],newc],0)

            if constraint_type == 'cluster':
                selected_nodes.add(cluster[a])
                selected_nodes.add(cluster[b])

            if len(components) == 1:
                break

        if len(components) == 1:
            real_adj_mat[components[0, 1], components[0, 0]] = 1
        real_adj_mat += real_adj_mat.T

        if num_cities != 200:
            max_iter = None
        tour = construct_tsp_from_mst(adj_mat, real_adj_mat, dists, points, constraint_type, constraint)
        tsp_solver = TSP_2opt(points, constraint_type, constraint)
        solved_tour, _ = tsp_solver.solve_2opt(tour, max_iter)

        # def has_duplicates(l):
        #     existing = []
        #     for item in l:
        #         if item in existing:
        #             return True
        #         existing.append(item)
        #     return False

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


if __name__ == "__main__":
    # Load test data
    data = np.load("data_20.npz")
    points = data['points']
    dists = data['dists']
    constraint = data['constraint']
    constraint_type = data['constraint_type'].item()
    model_latent = torch.load("model_latent_20.pt")