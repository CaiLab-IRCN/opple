'''
    Description: this module is for utility funcs used by and shared across external models
'''

import torch
import numpy as np

def coordshift(obj, cam):
    new_obj = []
    new_obj_transformed = []
    origin = torch.zeros(cam.shape, device = cam.device)
    origin[:, 4]=1
    origin[:, 5]=1
    origin[:, 7]=1

    new_obj_transformed = torch.ones([obj.shape[0], obj.shape[1], 4], device = cam.device)
    
    mat = camera_move_mat_pixel(origin, cam, device=cam.device)
    new_obj_transformed[:, :, 0:3] = obj
    new_obj = torch.matmul(mat[:,None,:,:], new_obj_transformed[:,:,:,None])[:,:,:3,0]
    return new_obj

def coordshift_pose(obj, cam):
    new_obj = []
    new_obj_transformed = []
    
    camposes = cam[:, 3:]
    new_obj_transformed = torch.ones(obj.shape, device = cam.device)
    for k in range(obj.shape[1], 2):
        new_obj_transformed[:, k] = obj[:, k]*camposes[:,k] + obj[:, k+1]*camposes[:,k+1]
        new_obj_transformed[:, k+1] = obj[:, k+1]*camposes[:,k] - obj[:, k]*camposes[:,k+1]
    new_obj = new_obj_transformed
    return new_obj

def label_maps_to_colour(masks):
    colors = np.asarray(
        [[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0], [0.8,0.6,0.0]]) # ['r', 'g', 'b', 'y'] # to change
    imgs = colors[np.argmax(masks, 1)]
    return imgs

def camera_move_mat_pixel(cam1, cam2, device):
    # Calculating the transformation of a pixel in camera's view
    # when the camera moves from position cam1 to cam2
    # cam1 and cam2 are coded as 
    # [x,y,z, cos(yaw), sin(yaw), cos(roll), sin(roll), cos(pitch), sin(pitch)]
    # cam1 and cam2 are in size of batch by 9
    
    # https://www.wikiwand.com/en/Rotation_matrix
    # https://math.stackexchange.com/questions/709622/relative-camera-matrix-pose-from-global-camera-matrixes

    ## camera corrdinates: x - right, y - away , z - upwards
    ## required coordinates: x - right, y - down, z - away

    def _ry(a,b):
        # a = cos(pitch), b = sin(pitch)
        return torch.stack([torch.stack([a, torch.zeros_like(a, device=device), -1*b], 1),
                            torch.stack([torch.zeros_like(a, device=device), torch.ones_like(a, device=device),
                                         torch.zeros_like(a, device=device)], 1),
                            torch.stack([b, torch.zeros_like(a, device=device), a],1)], 1)
    

    def _rz(a,b):
        # a = cos(yaw), b = sin(yaw)
        return torch.stack([torch.stack([a, b, torch.zeros_like(a, device=device)], 1),
                            torch.stack([-b, a, torch.zeros_like(a, device=device)], 1),
                            torch.stack([torch.zeros_like(a, device=device), torch.zeros_like(a, device=device),
                                         torch.ones_like(a, device=device)], 1)], 1)

    def _rx(a,b):
        # a = cos(roll), b = sin(roll)
        return torch.stack([torch.stack([torch.ones_like(a, device=device), torch.zeros_like(a, device=device),
                                         torch.zeros_like(a, device=device)], 1),
                            torch.stack([torch.zeros_like(a, device=device), a, b], 1),
                            torch.stack([torch.zeros_like(a, device=device), -1*b, a], 1)], 1)
    batch_size = cam1.shape[0]

    R1x = _rx(torch.ones(batch_size, device=device), torch.zeros(batch_size, device=device))
    R2x = R1x
    R1y = _ry(cam1[:,5], cam1[:,6])
    R2y = _ry(cam2[:,5], cam2[:,6])
    R1z = _rz(cam1[:,3], cam1[:,4])
    R2z = _rz(cam2[:,3], cam2[:,4])

    ## x-y plane rotation by -90 degrees 
    r90 = torch.eye(4, device=device).repeat(batch_size, 1, 1) 
    r90[:, :3, :3] = _rz(torch.zeros(batch_size, device=device),
                         torch.ones(batch_size, device=device))
    
    R1 = torch.eye(4, device=device).repeat(batch_size, 1, 1)
    R1[:, :3, :3] = R1z@R1y@R1x
    R2 = torch.eye(4, device=device).repeat(batch_size, 1, 1)
    R2[:, :3, :3] = R2z@R2y@R2x

    T = torch.zeros((4,4), device=device).repeat(batch_size, 1, 1) 
    T[:, :3, 3] = cam1[:,:3] - cam2[:,:3] 
    T = r90.transpose(1,2)@R2@T

    R = r90.transpose(1,2)@R2@R1.transpose(1,2)@r90 + T

    return R

