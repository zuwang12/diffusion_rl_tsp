import scipy
import numpy as np
import cv2
import math
from tqdm import tqdm

import torch
import torch.nn.functional as F
from model.diffusion import GaussianDiffusion

class TSP_2opt():
    def __init__(self, points):
        self.dist_mat = scipy.spatial.distance_matrix(points, points)
    
    def evaluate(self, route):
        total_cost = 0
        for i in range(len(route)-1):
            total_cost += self.dist_mat[route[i],route[i+1]]
        return total_cost

    def solve_2opt(self, route):
        assert route[0] == route[-1], 'Tour is not a cycle'

        best = route
        improved = True
        steps = 0
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)):
                    if j-i == 1: continue # changes nothing, skip then
                    new_route = route[:]
                    new_route[i:j] = route[j-1:i-1:-1] # this is the 2woptSwap
                    if self.evaluate(new_route) < self.evaluate(best):
                        best = new_route
                        steps += 1
                        improved = True
            route = best
        return best, steps
    
    def seed_solver(self, routes):
        best = self.solve_2opt(routes[0])
        for i in range(1, len(routes)):
            result = self.solve_2opt(routes[i])
            if self.evaluate(result) < self.evaluate(best):
                best = result
        return best
    
def runlat(model, unet, STEPS, batch_size, device):
    opt = torch.optim.Adam(model.parameters(), lr=1, betas=(0, 0.9))
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor=0.1, total_iters=1000)
    diffusion = GaussianDiffusion(T=1000, schedule='linear')
    # model.latent.data=temp

    steps = STEPS
    for i in tqdm(range(steps)):
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