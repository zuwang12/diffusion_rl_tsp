import torch
import numpy as np
import cv2
from utils import draw_tour

class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, img_size, constraint_type='basic', show_position=False, point_radius=1, point_color=1, point_circle=True, line_thickness=2, line_color=0.5, box_color=0.75, max_points=100):
        self.data_file = data_file
        self.img_size = img_size
        self.point_radius = point_radius
        self.point_color = point_color
        self.point_circle = point_circle
        self.line_thickness = line_thickness
        self.line_color = line_color
        self.box_color = box_color
        self.max_points = max_points
        self.constraint_type = constraint_type
        self.show_position = show_position
        
        self.file_lines = open(data_file).read().splitlines()
        print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')
        
    def __len__(self):
        return len(self.file_lines)
    
    def rasterize(self, idx):
        # Select sample
        line = self.file_lines[idx].strip()

        # Extract points
        points = line.split(' output ')[0].split(' ')
        points = np.array([[float(points[i]), float(points[i+1])] for i in range(0, len(points), 2)])
        
        # Extract tour
        tour = line.split(' output ')[1].split(' ')
        tour = np.array([int(t) for t in tour])
        
        constraint = None
        if self.constraint_type != 'basic':
            # Extract constraint if not basic type
            constraint = line.split(' output ')[2].split(' ')
            constraint = np.array([float(t) for t in constraint])
        
        # Draw the image based on the constraint type
        if self.constraint_type == 'box':        
            img = draw_tour(tour=tour, points=points, box=constraint, img_size=self.img_size, 
                            line_color=self.line_color, line_thickness=self.line_thickness,
                            point_color=self.point_color, point_circle=self.point_circle, point_radius=self.point_radius,
                            show_position=self.show_position, box_color=self.box_color)
        else:
            img = draw_tour(tour=tour, points=points, img_size=self.img_size, 
                            line_color=self.line_color, line_thickness=self.line_thickness,
                            point_color=self.point_color, point_circle=self.point_circle, point_radius=self.point_radius,
                            show_position=self.show_position, box_color=self.box_color)
            
        return img, points, tour, constraint

    def __getitem__(self, idx):
        img, points, tour, constraint = self.rasterize(idx)
        if self.constraint_type == 'basic':
            return img[np.newaxis, :, :], points, tour, idx, 0
        else:
            return img[np.newaxis, :, :], points, tour, idx, constraint