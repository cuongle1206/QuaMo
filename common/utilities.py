import os
import time
import argparse
import random
import numpy as np
import torch
import smplx
from pytorch3d import transforms

def get_parse():
    parser = argparse.ArgumentParser(description='Quaternion Differential Equation')
    parser.add_argument("--dataset", type=str, default="h36m")
    parser.add_argument("--exp", type=str, default="full", help="select 'ablation' or 'full'")
    parser.add_argument("--input", type=str, default="trace", help="select between 'trace' and 'hmr2'")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--warm", type=int, default=4)
    parser.add_argument("--bsize", type=int, default=64)
    parser.add_argument("--lrate", type=float, default=5e-4)
    parser.add_argument("--lstep", nargs='+', type=int, default=[20,30])
    parser.add_argument("--clip", type=float, default=0.8)
    parser.add_argument("--wsize", type=int, default=100, help="window size")
    parser.add_argument("--rotation", type=str, default="quaternion")
    parser.add_argument("--exact", action="store_false")
    parser.add_argument("--second", action="store_false")
    parser.add_argument("--lam", type=float, default=1e-2)
    return parser

def print_log(str, print_time=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        str = "[ " + localtime + ' ] ' + str
    print(str)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    
def load_smpl_models(project_path, device):
    # types = ["neutral", "female", "male"]
    types = ["neutral"]
    smpl_models = {}
    for gender in types:
        smpl_models[gender] = smplx.build_layer(
            model_path = f"{project_path}/smpl_models",
            model_type = "smpl",
            gender = gender,
            num_betas = 10,
            dtype = torch.float64 
            ).to(device)
    return smpl_models

def SMPL_forward(smpl_layer, betas, thetas, rep='quaternion'):
    N, T, _, _      = thetas.shape
    
    if rep == 'quaternion':
        global_orient   = transforms.quaternion_to_matrix(thetas[:,:,:1])
        body_pose       = transforms.quaternion_to_matrix(thetas[:,:,1:])
    elif rep == 'axis':
        global_orient   = transforms.axis_angle_to_matrix(thetas[:,:,:1])
        body_pose       = transforms.axis_angle_to_matrix(thetas[:,:,1:])
    elif rep == 'ZXY' or rep == 'XYZ':
        global_orient   = transforms.euler_angles_to_matrix(thetas[:,:,:1], rep)
        body_pose       = transforms.euler_angles_to_matrix(thetas[:,:,1:], rep)
    else:
        raise ValueError("Invalid rotation")
    
    smpl = smpl_layer(betas=betas.reshape(-1,10),
                      global_orient = global_orient.reshape(-1,3,3),
                      body_pose = body_pose.reshape(-1,23,3,3),
                      transl = None,
                      return_verts = True)
    return smpl.vertices.reshape(N,T,-1,3), smpl.joints.reshape(N,T,-1,3)

def SMPL_regression(mesh, regressor):
    p3d     = torch.einsum('ntvc,jv -> ntjc', mesh, regressor) # mesh -> 17j
    p3d_ra  = p3d - p3d[...,:1,:] # since the root is not at origin
    mesh_ra = mesh - p3d[...,:1,:]
    return mesh_ra, p3d_ra

def SMPL_regression_coco(mesh, regressor):
    p3d     = torch.einsum('ntvc,jv -> ntjc', mesh, regressor) # mesh -> 17j
    root    = 0.5*(p3d[...,11,None,:] + p3d[...,12,None,:])
    p3d_17j = torch.cat((root, p3d), dim=-2)
    p3d_ra  = p3d_17j - p3d_17j[...,:1,:] # since the root is not at origin
    mesh_ra = mesh - p3d_17j[...,:1,:]
    return mesh_ra, p3d_ra

def build_intrinsics(params):
    N       = params.shape[0]
    fx, fy, cx, cy = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    intrinsics = torch.zeros((N, 3, 3), device=params.device, dtype=params.dtype)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy
    intrinsics[:, 2, 2] = 1.0
    return intrinsics

def transform_to_quaternion(axis_angle): # N, T, 24, 3
    rot_mat = transforms.axis_angle_to_matrix(axis_angle) # N, T, 24, 3, 3
    quaternion = transforms.matrix_to_quaternion(rot_mat) 
    return quaternion

def transform_to_euler(axis_angle, rep):
    rot_mat = transforms.axis_angle_to_matrix(axis_angle) # N, T, 24, 3, 3
    euler = transforms.matrix_to_euler_angles(rot_mat, rep) 
    return euler