import numpy as np
import torch
from utils import TSP_2opt
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
    
    def _fn(points, model_latent, dists=None, constraint_matrix=None):
        penalty = 0
        # batch_size = 1
        # model_latent = torch.randn(batch_size,points.shape[0],points.shape[0])
        adj_mat = normalize((model_latent)).detach().cpu().numpy()[0] # model_latent : [1, 50, 50] -> adj_mat : (50, 50)
        adj_mat = adj_mat+adj_mat.T

        components = np.zeros((adj_mat.shape[0],2)).astype(int) # (50, 2)
        components[:] = np.arange(adj_mat.shape[0])[...,None] # (50, 1) | [[1], [2], ... , [49]]
        real_adj_mat = np.zeros_like(adj_mat) # (50, 50) 
        np.seterr(divide='ignore', invalid='ignore') # TODO: need to set error option globally
        for edge in (-adj_mat/dists).flatten().argsort(): # [1715,  784, 1335, ..., 1326, 1224, 2499]) | 실제 거리(dists) 대비 adj_mat값이 가장 높은 순으로 iter
            a,b = edge//adj_mat.shape[0],edge%adj_mat.shape[0] # (34, 15)
            if not (a in components and b in components): continue
            if constraint_matrix is not None:
                if constraint_matrix[a][b] == 1:continue
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
            if set(n).issubset(set(tour)):
                tmp_node = random.choice([x for x in range(adj_mat.shape[0]) if x not in tour]) # 가능한 연결수단이 없으면 남은 노드중에 선택하고, penalty 추가
                tour.append(tmp_node)
                penalty += 10
            else:
                tour.append(n.max())
            if len(tour) == adj_mat.shape[0]: # finalize adding root node
                tour.append(tour[0])
                break

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
        # assert not has_duplicates(solved_tour[:-1]), 'Tour not Hamiltonian' # constraint 조건에서는 penalty 받는 형태로.. TODO: 더 좋은 방법 없을까?

        # gt_cost = tsp_solver.evaluate([i-1 for i in gt_tour]) # TODO: one times
        solved_cost = tsp_solver.evaluate(solved_tour)
        # print('solved cost : ', solved_cost)
        return -np.array([solved_cost]), {
            'solved_tour' : np.array(solved_tour)+1, 
            # 'points' : points, 
            # 'gt_cost' : gt_cost,
            'solved_cost' : solved_cost + penalty,
            }
    return _fn