import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time

def get_rays(height, width, intrinsics, Rcw, Tcw):
    
    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    Rcw: Rotation matrix of shape (3,3) from camera to world coordinates.
    Tcw: Translation vector of shape (3,1) that transforms

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that desipte that all ray share the same origin, here we ask you to return 
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder
    
    #############################  TODO 2.1 BEGIN  ##########################  
   
    u, v= np.meshgrid(np.arange(width), np.arange(height))
   
    # pixel_coords = np.column_stack((u, v, np.ones_like(u)))
    pixel_coords = np.stack((u, v, np.ones_like(u)), axis = -1)
    
    pixel_coords = pixel_coords.reshape(-1,3).T
    
    pixel_coords = torch.tensor(pixel_coords, dtype = torch.float32)
  
    
    ray_origins = Tcw.repeat(height, width, 1)
    ray_directions = ((Rcw @ torch.inverse(intrinsics) @ pixel_coords).T).reshape(height, width, 3)

     
    #############################  TODO 2.1 END  ############################
    return ray_origins, ray_directions

def stratified_sampling(ray_origins, ray_directions, near, far, samples):

    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.
  
    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """

    #############################  TODO 2.2 BEGIN  ############################
    
    t_values = torch.linspace(near, far, samples)
    height, width = ray_origins.shape[:2]
    depth_points = torch.zeros((height, width, samples)) 
    ray_points = torch.zeros((height, width, samples, 3))  
     
    for i in range(height): 
      for j in range(width): 
        origin = ray_origins[i][j]
        dir = ray_directions[i][j]
        for k,t in enumerate(t_values): 
          ray_points[i][j][k] = origin + t * dir
          depth_points[i][j][k] = t


    #############################  TODO 2.2 END  ############################
    return ray_points, depth_points
    
class nerf_model(nn.Module):
    
    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper. 
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        #############################  TODO 2.3 BEGIN  ############################
        #Linear layers 
        self.linear1 = nn.Linear((6*num_x_frequencies)+3 , filter_size)
        self.linear2 = nn.Linear(in_features = filter_size, out_features = filter_size)
        self.linear3 = nn.Linear(in_features = filter_size, out_features = filter_size) 
        self.linear4 = nn.Linear(in_features = filter_size, out_features = filter_size)
        self.linear5 = nn.Linear(in_features = filter_size, out_features = filter_size)
        self.linear6 = nn.Linear(in_features = filter_size+6*num_x_frequencies+3, out_features = filter_size) 
        self.linear7 = nn.Linear(in_features = filter_size, out_features = filter_size) 
        self.linear8 = nn.Linear(in_features = filter_size, out_features = filter_size)
        self.linear9 = nn.Linear(in_features = filter_size, out_features = 1)
        self.linear10 = nn.Linear(in_features = filter_size, out_features = filter_size) 
        self.linear11 = nn.Linear(in_features = filter_size+6*num_d_frequencies+3, out_features = 128) 
        self.linear12 = nn.Linear(in_features = 128, out_features = 3) 

        #############################  TODO 2.3 END  ############################


    def forward(self, x, d):
        #############################  TODO 2.3 BEGIN  ############################
        output = F.relu(self.linear1(x))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = F.relu(self.linear4(output))
        output = F.relu(self.linear5(output))
        output = F.relu(self.linear6(torch.cat([output, x], dim = 1)))
        output = F.relu(self.linear7(output))
        output = F.relu(self.linear8(output))
        sigma = self.linear9(output)
        output = self.linear10(output)
        output = F.relu(self.linear11(torch.cat([output,d], dim = 1)))
        rgb = torch.sigmoid(self.linear12(output))
        #############################  TODO 2.3 END  ############################
        return rgb, sigma

def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):
    
    def positional_encoding(x, num_frequencies=6, incl_input=True):
        
      """
      Apply positional encoding to the input.
      
      Args:
      x (torch.Tensor): Input tensor to be positionally encoded. 
        The dimension of x is [N, D], where N is the number of input coordinates,
        and D is the dimension of the input coordinate.
      num_frequencies (optional, int): The number of frequencies used in
      the positional encoding (default: 6).
      incl_input (optional, bool): If True, concatenate the input with the 
          computed positional encoding (default: True).

      Returns:
      (torch.Tensor): Positional encoding of the input tensor. 
      """
      
      results = []
      if incl_input:
          results.append(x)
      #############################  TODO 1(a) BEGIN  ############################
      # encode input tensor and append the encoded tensor to the list of results.

      #try just appenidng in for loop
      for i in range(num_frequencies): 
          encoded_sin = torch.sin(x * (2 ** i) * torch.pi)
          results.append(encoded_sin)

          encoded_cos = torch.cos(x * (2 ** i) * torch.pi)
          results.append(encoded_cos)
      
      #############################  TODO 1(a) END  ##############################
      return torch.cat(results, dim=-1)


    def get_chunks(inputs, chunksize = 2**15):
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]
    
    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before 
    dividing them into chunks, as well as normalizing and populating the directions.
    """
    #############################  TODO 2.3 BEGIN  ############################
    
    #normalize 
    ray_directions_norm = ray_directions/torch.linalg.norm(ray_directions, dim = -1, keepdim = True)
    #populate
    ray_directions_pop = ray_directions_norm.unsqueeze(2).repeat(1,1, ray_points.shape[2],1)
    #flatten
    ray_directions_flat = ray_directions_pop.reshape(-1,3)
    #positional encoding 
    ray_directions_encoded = positional_encoding(ray_directions_flat, num_d_frequencies)
    #call get_chunks
    ray_directions_batches = get_chunks(ray_directions_encoded)
    #repeat the same for points
    ray_points_flat = ray_points.reshape(-1,3)
    ray_points_encoded = positional_encoding(ray_points_flat, num_x_frequencies)
    ray_points_batches = get_chunks(ray_points_encoded)

    #############################  TODO 2.3 END  ############################

    return ray_points_batches, ray_directions_batches

def volumetric_rendering(rgb, s, depth_points):

    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    sigma: Volume density at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
  
    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """
    
    #############################  TODO 2.4 BEGIN  ############################
    height, width, samples = depth_points.shape
      
    delta = torch.zeros((height, width, samples))
    delta[:, :, :-1] = depth_points[:, :, 1:] - depth_points[:, :, :-1]
    delta[:, :, -1] = 1e9

    s = F.relu(s)
    # singleT = -s*delta
    singleT = -s * delta

    #initialize T first????
    T = torch.ones_like(singleT)
    T[:, :, 1:] = torch.exp(torch.cumsum(singleT[:, :, :-1], dim = -1))
    # T_prod = torch.cumprod(torch.exp(singleT), dim=-1)
    # T = torch.cat([torch.ones((height, width, 1)), T_prod], dim=-1)
    # T[:, :, -1] = torch.exp(-1 * torch.sum(singleT[:, :, :-1], dim=-1))

    C = T.unsqueeze(-1) * (1 - torch.exp(singleT)).unsqueeze(-1) * rgb
    rec_image = torch.sum(C, dim=2)

    #############################  TODO 2.4 END  ############################

    return rec_image

def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):
    
    #############################  TODO 2.5 BEGIN  ############################

    #compute all the rays from the image
    R =  torch.tensor([row[:3] for row in pose[:3]])
    
    T = torch.tensor([row[3] for row in pose[:3]])
    ray_origins, ray_directions = get_rays(height, width, intrinsics, R, T)

    #sample the points from the rays
    ray_points, depth_points = stratified_sampling(ray_origins, ray_directions, near, far, samples)

    #divide data into batches to avoid memory errors
    ray_point_batches, ray_direction_batches = get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies)

    #forward pass the batches and concatenate the outputs at the end
    





    # Apply volumetric rendering to obtain the reconstructed image


    #############################  TODO 2.5 END  ############################

    return rec_image