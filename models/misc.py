import torch
from pytorch3d import transforms

def axis_angle_to_quaternion(axis_angle: torch.Tensor, scale: float) -> torch.Tensor:
    angles = scale * torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    sin_half_angles_over_angles = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
    return torch.cat(
        [torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], dim=-1
    )

def get_quaternion_error(q, qtar):
    q_error = transforms.quaternion_multiply(qtar, transforms.quaternion_invert(q)) # hemisphere alignement
    return q_error[...,1:]

def masking(x, jnt=[7,8,10,11,20,21,22,23]):
    x[:,jnt] = x[:,jnt].clone() * 0.
    return x

def masking_coco(x, jnt=[7,8,10,11,15,20,21,22,23]):
    x[:,jnt] = x[:,jnt].clone() * 0.
    return x


"""
Quaternion
PyTorch adaptation from rowan https://github.com/glotzerlab/rowan
"""

def quat_exp(q: torch.Tensor):
    
    expo    = torch.empty(q.shape).to(q.device)
    norms   = torch.norm(q[..., 1:], dim=-1)
    e       = torch.exp(q[..., 0])
    expo[..., 0] = e * torch.cos(norms)
    norm_zero   = torch.isclose(norms, torch.zeros_like(norms))
    not_zero    = torch.logical_not(norm_zero)
    if torch.any(not_zero):
        expo[..., 1:][not_zero] = (
            e[not_zero].unsqueeze(-1)
            * (q[..., 1:][not_zero] / norms[not_zero].unsqueeze(-1))
            * torch.sin(norms)[not_zero].unsqueeze(-1)
        )
        if torch.any(norm_zero):
            expo[..., 1:][norm_zero] = 0
    else:
        expo[..., 1:] = 0

    return expo

def quat_log(q: torch.Tensor):
    # N,T,V,C     = q.shape
    log         = torch.empty(q.shape).to(q.device)
    
    q_norms     = torch.norm(q, dim=-1)
    q_norm_zero = torch.isclose(q_norms, torch.zeros_like(q_norms))
    q_not_zero  = torch.logical_not(q_norm_zero)
    
    v_norms     = torch.norm(q[..., 1:], dim=-1)
    v_norm_zero = torch.isclose(v_norms, torch.zeros_like(v_norms))
    v_not_zero  = torch.logical_not(v_norm_zero)
    
    if torch.any(q_not_zero):
        if torch.any(q_norm_zero):
            log[..., 0][q_norm_zero] = -torch.inf
        log[..., 0][q_not_zero] = torch.log(q_norms[q_not_zero])
    else:
        log[..., 0] = -torch.inf
        
    if torch.any(v_not_zero):
        prefactor = torch.empty(q[..., 1:][v_not_zero].shape)
        prefactor = q[..., 1:][v_not_zero] / v_norms[v_not_zero].unsqueeze(-1)

        inv_cos = torch.empty(v_norms[v_not_zero].shape)
        inv_cos = torch.arccos(q[..., 0][v_not_zero] / q_norms[v_not_zero])

        if torch.any(v_norm_zero):
            log[..., 1:][v_norm_zero] = 0
        log[..., 1:][v_not_zero] = prefactor * inv_cos.unsqueeze(-1)
    else:
        log[..., 1:] = 0
    
    return log

def slerp(q0: torch.Tensor, q1: torch.Tensor, t, short_path=True):
    
    # Ensure that we turn the short way around
    if short_path:
        cos_half_angle = torch.squeeze(q0.unsqueeze(-2) @ q1.unsqueeze(-1), dim=[-1, -2])
        flip        = cos_half_angle < 0
        q1[flip]    *= -1
        
    rel_quat    = transforms.quaternion_raw_multiply(transforms.quaternion_invert(q0), q1)
    slerp_t     = transforms.quaternion_raw_multiply(q0, quat_exp(quat_log(rel_quat) * t))
    return slerp_t