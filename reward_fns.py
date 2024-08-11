import numpy as np
import torch
from utils import TSP_2opt, do_intersect, construct_tsp_from_mst, check_for_intersection
import random

def tsp():
    def normalize(cost, entropy_reg=0.1, n_iters=20, eps=1e-6):
        # Cost matrix is exp(-lambda*C)
        cost_matrix = -entropy_reg * cost # 0.1 * [1, 50, 50] (latent)
            
        cost_matrix -= torch.eye(cost_matrix.shape[-1], device=cost_matrix.device)*100000 # COST = COST - 100000*I
        cost_matrix = cost_matrix - torch.logsumexp(cost_matrix, dim=-1, keepdim=True)
        assignment_mat = torch.exp(cost_matrix)
        
        return assignment_mat # [1, 50, 50] (adj_mat)
    
    def _fn(points, model_latent, dists=None):
        # batch_size = 1
        # model_latent = torch.randn(batch_size,points.shape[0],points.shape[0])
        adj_mat = normalize((model_latent)).detach().cpu().numpy()[0] # model_latent : [1, 50, 50] -> adj_mat : (50, 50)
        adj_mat = adj_mat+adj_mat.T

        components = np.zeros((adj_mat.shape[0],2)).astype(int) # (50, 2)
        components[:] = np.arange(adj_mat.shape[0])[...,None] # (50, 1) | [[1], [2], ... , [49]] -> broadcasting
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
        solved_tour, _ = tsp_solver.solve_2opt(tour)

        def has_duplicates(l):
            existing = []
            for item in l:
                if item in existing:
                    return True
                existing.append(item)
            return False

        assert solved_tour[-1] == solved_tour[0], 'Tour not a cycle'
        assert not has_duplicates(solved_tour[:-1]), 'Tour not Hamiltonian'

        # gt_cost = tsp_solver.evaluate([i-1 for i in gt_tour]) # TODO: one times
        solved_cost = tsp_solver.evaluate(solved_tour)
        # print('solved cost : ', solved_cost)
        return -np.array([solved_cost]), {
            'solved_tour' : solved_tour, 
            # 'points' : points, 
            # 'gt_cost' : gt_cost,
                'solved_cost' : solved_cost,}
    return _fn


def tsp_constraint():
    def normalize(cost, entropy_reg=0.1, n_iters=20, eps=1e-6):
        # Cost matrix is exp(-lambda*C)
        cost_matrix = -entropy_reg * cost # 0.1 * [1, 50, 50] (latent)
            
        cost_matrix -= torch.eye(cost_matrix.shape[-1], device=cost_matrix.device)*100000 # COST = COST - 100000*I
        cost_matrix = cost_matrix - torch.logsumexp(cost_matrix, dim=-1, keepdim=True)
        assignment_mat = torch.exp(cost_matrix)
        
        return assignment_mat # [1, 50, 50] (adj_mat)
    
    def _fn(points, model_latent, dists, constraint_type = None, constraint = None):
        if constraint_type == 'box':
            constraint_matrix = constraint
        elif constraint_type == 'path':
            path = constraint
        elif constraint_type == 'cluster':
            cluster = constraint
            
        penalty = 0
        # batch_size = 1
        # model_latent = torch.randn(batch_size,points.shape[0],points.shape[0])
        adj_mat = normalize((model_latent)).detach().cpu().numpy()[0] # model_latent : [1, 50, 50] -> adj_mat : (50, 50)
        adj_mat = adj_mat+adj_mat.T

        components = np.zeros((adj_mat.shape[0],2)).astype(int) # (50, 2)
        components[:] = np.arange(adj_mat.shape[0])[...,None] # (50, 1) | [[1], [2], ... , [49]]
        real_adj_mat = np.zeros_like(adj_mat) # (50, 50) 
        np.seterr(divide='ignore', invalid='ignore') # TODO: need to set error option globally
        
        # Ensure that mandatory paths are connected
        if constraint_type == 'path':
        # if path is not None:
            for i in range(0, len(path), 2):
                a, b = int(path[i]), int(path[i + 1])
                real_adj_mat[a, b] = 1  # Connect mandatory paths

                ca = np.nonzero((components == a).sum(1))[0][0]
                cb = np.nonzero((components == b).sum(1))[0][0]
                cca = sorted(components[ca], key=lambda x: x == a)
                ccb = sorted(components[cb], key=lambda x: x == b)
                newc = np.array([[cca[0], ccb[0]]])
                m, M = min(ca, cb), max(ca, cb)
                components = np.concatenate([components[:m], components[m + 1:M], components[M + 1:], newc], 0)

        if constraint_type == 'cluster':
            selected_nodes = set()  # Track selected clusters
        
        for edge in (-adj_mat/dists).flatten().argsort(): # [1715,  784, 1335, ..., 1326, 1224, 2499]) | 실제 거리(dists) 대비 adj_mat값이 가장 높은 순으로 iter
            a,b = edge//adj_mat.shape[0],edge%adj_mat.shape[0] # (34, 15)
            if a==b:
                continue
            if not (a in components and b in components):
                continue
            
            # TODO: 공통 ? for box?
            if check_for_intersection(a, b, real_adj_mat, points): 
                continue
            
            if constraint_type == 'box':
                if constraint_matrix[a][b] == 1:
                    continue
                
            # Ensure only one node per cluster is selected
            if constraint_type == 'cluster':
                if cluster[a] in selected_nodes and cluster[b] in selected_nodes:
                    continue
                
            ca = np.nonzero((components==a).sum(1))[0][0] # 34
            cb = np.nonzero((components==b).sum(1))[0][0] # 15
            if ca==cb: 
                continue
            cca = sorted(components[ca],key=lambda x:x==a) # [34, 34]
            ccb = sorted(components[cb],key=lambda x:x==b) # [15, 15]
            newc = np.array([[cca[0],ccb[0]]]) # [34, 15]
            m,M = min(ca,cb),max(ca,cb) # (15, 34)
            real_adj_mat[a,b] = 1 # 연결됨
            components = np.concatenate([components[:m],components[m+1:M],components[M+1:],newc],0) # (49, 2)
            
            # Mark the clusters as selected
            if constraint_type == 'cluster':
                selected_nodes.add(cluster[a])
                selected_nodes.add(cluster[b])
            
            if len(components)==1:
                break
        
        if len(components)==1:
            real_adj_mat[components[0,1],components[0,0]] = 1 # 마지막 연결
        real_adj_mat += real_adj_mat.T # make symmetric matrix

        tour = construct_tsp_from_mst(adj_mat, real_adj_mat, dists, points, constraint_type, constraint)

        # Refine using 2-opt
        tsp_solver = TSP_2opt(points, constraint_type, constraint)
        solved_tour, _ = tsp_solver.solve_2opt(tour)

        def has_duplicates(l):
            existing = []
            for item in l:
                if item in existing:
                    return True
                existing.append(item)
            return False

        assert solved_tour[-1] == solved_tour[0], 'Tour not a cycle'
        # assert not has_duplicates(solved_tour[:-1]), 'Tour not Hamiltonian' # constraint 조건에서는 penalty 받는 형태로.. TODO: 더 좋은 방법 없을까?

        # gt_cost = tsp_solver.evaluate([i-1 for i in gt_tour]) # TODO: one times
        solved_cost = tsp_solver.evaluate(solved_tour) # TODO: hard / soft 구분
        # Calculate the penalty for constraints
        penalty_const = 10  # Define a penalty constant
        penalty_count = tsp_solver.count_constraints(solved_tour)  # Count the number of constraint violations
        penalty = penalty_count * penalty_const  # Calculate the penalty
        total_cost = solved_cost + penalty  # Calculate the total cost including penalty
        
        return -np.array([solved_cost]), {
            'solved_tour' : np.array(solved_tour)+1, 
            # 'points' : points, 
            # 'gt_cost' : gt_cost,
            'solved_cost' : total_cost,
            'basic_cost' : solved_cost,
            'penalty_count' : penalty_count,
            }
    return _fn