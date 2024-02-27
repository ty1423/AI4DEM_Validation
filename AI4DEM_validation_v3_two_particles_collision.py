import os
import numpy as np 
import pandas as pd
import time 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print("Using GPU:", is_gpu)

# Function to generate a structured grid
def create_grid(domain_size, cell_size):
    num_cells = int(domain_size / cell_size)
    grid = np.zeros((num_cells, num_cells), dtype=int)
    return grid

# Input Parameters
domain_size = 20  # Size of the square domain
cell_size = 1   # Cell size and particle radius
simulation_time = 1
kn = 500  # Normal stiffness of the spring
dn = 0.5  # Normal damping coefficient
particle_mass = 1
K_graph = 28.2*10000*1
S_graph = K_graph * (cell_size / domain_size) ** 2


# Module 1: Domain discretisation and initial particle insertion
# Create grid
grid = create_grid(domain_size, cell_size)
grid_shape = grid.shape

# Generate particles
npt = int(domain_size ** 2)
x_grid = np.zeros((1, 1, grid_shape[0], grid_shape[1]))
y_grid = np.zeros((1, 1, grid_shape[0], grid_shape[1]))
vx_grid = np.zeros((1, 1, grid_shape[0], grid_shape[1]))
vy_grid = np.zeros((1, 1, grid_shape[0], grid_shape[1]))

# Insert particles
x_indices = [8,12]
y_indices = [10,10] # 80
for i, j in zip(x_indices, y_indices):
    x_grid[0, 0, j, i] = i * cell_size
    y_grid[0, 0, j, i] = j * cell_size
    
    
vx_grid[0, 0, 10, 8] = 1
vx_grid[0, 0, 10, 12] = -1



mask = np.where(x_grid != 0, 1, 0) 
print('Number of particles:', np.count_nonzero(mask))

# Define the AI4DEM model
class AI4DEM(nn.Module):
    """AI4DEM model for particle interaction detection and force calculation"""
    def __init__(self):
        super(AI4DEM, self).__init__()

    def detector(self, grid, i, j):
        """Detect neighboring particles and calculate the distance between them"""
        diff = grid - torch.roll(grid, shifts=(j - 2, i - 2), dims=(2, 3))
        return diff

    def forward(self, x_grid, y_grid, vx_grid, vy_grid, fx_grid, fy_grid, mask, d, kn, diffx, diffy, dt, input_shape, filter_size):
        cell_xold = x_grid / d
        cell_yold = y_grid / d   
        cell_xold = torch.round(cell_xold).long()
        cell_yold = torch.round(cell_yold).long()
        # calculate distance between the two particles 
        fx_grid = torch.zeros(input_shape, device=device) 
        fy_grid = torch.zeros(input_shape, device=device)
        for i in range(filter_size):
            for j in range(filter_size):
                diffx = self.detector(x_grid, i, j) # individual
                diffy = self.detector(y_grid, i, j) # individual
                dist = torch.sqrt(diffx**2 + diffy**2)   
                fx_grid =  fx_grid + torch.where(torch.lt(dist,2*d), kn * (dist - 2 * d ) * diffx / torch.maximum(eplis, dist), zeros) # individual                
                fy_grid =  fy_grid + torch.where(torch.lt(dist,2*d), kn * (dist - 2 * d ) * diffy / torch.maximum(eplis, dist), zeros) # individual       

        is_bottom_overlap = torch.gt(y_grid, 0.01) & torch.lt(y_grid, d) # Overlap with bottom wall
        is_top_overlap = torch.gt(y_grid,-d+domain_size ) & torch.lt(y_grid,domain_size) # Overlap with bottom wall      
        is_left_overlap = torch.gt(x_grid, 0.01) & torch.lt(x_grid, d) # Overlap with bottom wall
        is_right_overlap = torch.gt(x_grid,-d+domain_size ) & torch.lt(x_grid,domain_size) # Overlap with bottom wall
        
        is_bottom_overlap = is_bottom_overlap.to(device)
        is_top_overlap = is_top_overlap.to(device)
        is_left_overlap = is_left_overlap.to(device)
        is_right_overlap = is_right_overlap.to(device)
        fy_grid_boundary_bottom = kn * torch.where(is_bottom_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))*mask*(d-y_grid)
        fy_grid_boundary_top = -kn * torch.where(is_top_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))*mask*(-domain_size+y_grid+d)
        fx_grid_boundary_left = kn * torch.where(is_left_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))*mask*(d-x_grid)
        fx_grid_boundary_right = -kn * torch.where(is_right_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))*mask*(-domain_size+x_grid+d)
        vx_grid = vx_grid   - (dt / particle_mass)* ( fx_grid_boundary_left + fx_grid_boundary_right+fx_grid*mask) 
        vy_grid = vy_grid  - (dt / particle_mass) * ( 0 * particle_mass) * mask  + (dt / particle_mass)* ( fy_grid_boundary_top + fy_grid_boundary_bottom - fy_grid*mask) 
        # Update particle coordniates
        x_grid = x_grid + dt * vx_grid
        y_grid = y_grid + dt * vy_grid
        
        x_grid_merge = x_grid.clone()
        y_grid_merge = y_grid.clone()
        vx_grid_merge = vx_grid.clone()
        vy_grid_merge = vy_grid.clone()
        # update new index tensor 
        cell_x = x_grid / d 
        cell_y = y_grid / d     
        cell_x = torch.round(cell_x).long()
        cell_y = torch.round(cell_y).long()    
        # extract index (previous and new) from sparse index tensor (previous and new)
        cell_x = cell_x[cell_x!=0]
        cell_y = cell_y[cell_y!=0]         
        cell_xold = cell_xold[cell_xold!=0]
        cell_yold = cell_yold[cell_yold!=0]   
        # get rid of values at previous index 
        mask[0,0,cell_yold.long(), cell_xold.long()] = 0
        x_grid[0,0,cell_yold.long(),cell_xold.long()] = 0 
        y_grid[0,0,cell_yold.long(),cell_xold.long()] = 0 
        vx_grid[0,0,cell_yold.long(),cell_xold.long()] = 0 
        vy_grid[0,0,cell_yold.long(),cell_xold.long()] = 0 
        # update new values based on new index         
        mask[0,0,cell_y.long(), cell_x.long()] = 1
        x_grid[0,0,cell_y.long(), cell_x.long()] = x_grid_merge[0,0,cell_yold.long(),cell_xold.long()] 
        y_grid[0,0,cell_y.long(), cell_x.long()] = y_grid_merge[0,0,cell_yold.long(),cell_xold.long()] 
        vx_grid[0,0,cell_y.long(), cell_x.long()] = vx_grid_merge[0,0,cell_yold.long(),cell_xold.long()] 
        vy_grid[0,0,cell_y.long(), cell_x.long()] = vy_grid_merge[0,0,cell_yold.long(),cell_xold.long()]

        return x_grid, y_grid, vx_grid, vy_grid, mask

model = AI4DEM().to(device)

# Module 2: Contact detection and force calculation
t = 0
dt = 0.001  # 0.0001
ntime = 10000
# Convert np.array into torch.tensor and transfer it to GPU
filter_size = 5 
input_shape_global = (1, 1, grid_shape[0], grid_shape[1])

# Initialize tensors
diffx = torch.zeros(input_shape_global, device=device)
diffy = torch.zeros(input_shape_global, device=device)
zeros = torch.zeros(input_shape_global, device=device)
eplis = torch.ones(input_shape_global, device=device) * 1e-04
fx_grid = torch.zeros(input_shape_global, device=device)
fy_grid = torch.zeros(input_shape_global, device=device)
# vx_grid = torch.zeros(input_shape_global, device=device)
# vy_grid = torch.zeros(input_shape_global, device=device)

mask = torch.from_numpy(mask).float().to(device)
x_grid = torch.from_numpy(x_grid).float().to(device)
y_grid = torch.from_numpy(y_grid).float().to(device)

vx_grid = torch.from_numpy(vx_grid).float().to(device)
vy_grid = torch.from_numpy(vy_grid).float().to(device)

# Main simulation loop
start = time.time()
with torch.no_grad():
    for itime in range(1, ntime + 1):
        [x_grid, y_grid, vx_grid, vy_grid, mask] = model(x_grid, y_grid, vx_grid, vy_grid, fx_grid, fy_grid, mask, cell_size, kn, diffx, diffy, dt, input_shape_global, filter_size)
        print('Time step:', itime, 'Number of particles:', torch.count_nonzero(mask).item()) 
        
        if itime % 2 == 0:
            # Visualize particles
            xp = x_grid[x_grid!=0].cpu() 
            yp = y_grid[y_grid!=0].cpu() 
            
            plt.scatter(xp, yp, c=vx_grid[vx_grid!=0].cpu(), cmap='turbo', s=S_graph, vmin=-0.1, vmax=0.1)            
            cbar = plt.colorbar()
            cbar.set_label('$V_{p}$')
            ax = plt.gca()
            ax.set_xlim([0, domain_size])
            ax.set_ylim([0, domain_size])

            # Save visualization
            if itime < 10:
                save_name = "validation_two_two_particles_collision/"+str(itime)+".jpg"
            elif itime >= 10 and itime < 100:
                save_name = "validation_two_two_particles_collision/"+str(itime)+".jpg"
            elif itime >= 100 and itime < 1000:
                save_name = "validation_two_two_particles_collision/"+str(itime)+".jpg"
            elif itime >= 1000 and itime < 10000:
                save_name = "validation_two_two_particles_collision/"+str(itime)+".jpg"
            else:
                save_name = "validation_two_two_particles_collision/"+str(itime)+".jpg"
            plt.savefig(save_name, dpi=200, bbox_inches='tight')
            plt.close()

end = time.time()
print('Elapsed time:', end - start)
