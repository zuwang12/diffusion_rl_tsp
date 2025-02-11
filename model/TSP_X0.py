import torch
from torch import nn
import numpy as np
import cv2
from copy import deepcopy
from torchvision.utils import save_image

def normalize(cost, entropy_reg=0.1, n_iters=20, eps=1e-6):
    # Cost matrix is exp(-lambda*C)
    cost_matrix = -entropy_reg * cost # 0.1 * [1, N, N] (latent)
        
    cost_matrix -= torch.eye(cost_matrix.shape[-1], device=cost_matrix.device)*100000 # COST = COST - 100000*I
    cost_matrix = cost_matrix - torch.logsumexp(cost_matrix, dim=-1, keepdim=True)
    assignment_mat = torch.exp(cost_matrix)
    
    return assignment_mat # [1, N, N] (adj_mat)

class Model_x0(nn.Module):
    def __init__(self, batch_size, num_points, img_size, line_color, line_thickness, xT):
        super(Model_x0, self).__init__()
        
        # Latent variables (b,v,v) matrix
        self.latent = nn.Parameter(torch.randn(batch_size, num_points, num_points)) # (B, 50, 50)
        self.latent.requires_grad = True
        self.adj_mat = normalize(self.latent)
        
        self.num_points = num_points
        self.batch_size = batch_size
        self.img_size = img_size
        self.line_color = line_color
        self.line_thickness = line_thickness
        self.xT = xT

    def reset(self):
        nn.init.normal_(self.latent)
        # self.latent = nn.Parameter(torch.randn(self.batch_size, self.num_points, self.num_points)) # (B, 50, 50)
        # self.latent.requires_grad = True
    
    def get_adj_mat(self):
        return normalize(self.latent)

    def compute_edge_images(self, points, img_query):
        # Pre-compute edge images
        self.img_query = img_query
        num_nodes = points.shape[0]
        
        self.edge_images = torch.zeros((num_nodes, num_nodes, self.img_size, self.img_size), dtype=torch.float32, device=self.latent.device)

        scaled_points = ((self.img_size - 1) * points[:, ::-1]).astype(int)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_img = np.zeros((self.img_size, self.img_size), dtype=np.float16)
                cv2.line(edge_img, 
                        tuple(scaled_points[i]), 
                        tuple(scaled_points[j]), 
                        color=self.line_color, thickness=self.line_thickness)

                edge_tensor = torch.from_numpy(edge_img).float().to(self.latent.device)
                self.edge_images[i, j] = edge_tensor
                self.edge_images[j, i] = edge_tensor

    def encode(self, sampling=False):
        # Compute permutation matrix
        adj_mat = normalize(self.latent) # [1, N, N] -> [1, N, N]
        if sampling:
            adj_mat = normalize(deepcopy(self.latent)) #TODO: need to apply random
        self.adj_mat = adj_mat
        all_edges = self.edge_images.view(1,-1,self.img_size,self.img_size).to(adj_mat.device)
        img = all_edges * self.adj_mat.view(self.batch_size,-1,1,1) # [1, NxN, 64, 64] * [1, N, N] -> [1, NxN, 64, 64]
        img = torch.sum(img, dim=1, keepdims=True) # [1, NxN, 64, 64] -> [1, 1, 64, 64]
        
        img = 2*(img-0.5)               
        
        # Draw fixed points
        img[self.img_query.tile(self.batch_size,1,1,1) == 1] = 1
        
        return img
    
    def save_image(self, path):
        model_encode = deepcopy(torch.clamp(self.encode(), -1, 1).cpu().detach())
        model_encode -= model_encode.min()
        model_encode /= model_encode.max()
        save_image(model_encode[0,0,:,:], path)
        