import torch

L1Loss_avg = torch.nn.L1Loss()
L1Loss_sum = torch.nn.L1Loss(reduction="sum")
L2Loss_avg = torch.nn.MSELoss()
L2Loss_sum = torch.nn.MSELoss(reduction="sum")

def pose_l1_loss(predicted, target):
    assert predicted.shape == target.shape
    mask    = (target == 0).all(dim=(-2,-1))
    L1      = (torch.abs(predicted-target).sum(-1))[~mask]
    return L1.mean()

def trans_l1_loss(predicted, target):
    assert predicted.shape == target.shape
    mask    = (target == 0).all(dim=(-1))
    L1      = (torch.abs(predicted-target).sum(-1))[~mask]
    return L1.mean()

def proj_l1_loss(predicted, target):
    assert predicted.shape == target.shape
    mask    = (target == 0).all(dim=(-1))
    L1      = (torch.abs(predicted-target).sum(-1))[~mask]
    return L1.mean()

def smooth_pose_l1_loss(p_ra17, pgt_ra17):
    mask        = (pgt_ra17 == 0).all(dim=(-2,-1))
    p_2diff     = p_ra17[:,:-2,...] - 2*p_ra17[:,1:-1,...] + p_ra17[:,2:,...]
    pgt_2diff   = pgt_ra17[:,:-2,...] - 2*pgt_ra17[:,1:-1,...] + pgt_ra17[:,2:,...]
    L1          = (torch.abs(p_2diff-pgt_2diff).sum(-1))[~mask[:,1:-1]]
    return L1.mean()

def smooth_trans_l1_loss(t, tgt):
    mask        = (tgt == 0).all(dim=(-1))
    t_2diff     = t[:,:-2,...] - 2*t[:,1:-1,...] + t[:,2:,...]
    tgt_2diff   = tgt[:,:-2,...] - 2*tgt[:,1:-1,...] + tgt[:,2:,...]
    L1          = (torch.abs(t_2diff-tgt_2diff).sum(-1))[~mask[:,1:-1]]
    return L1.mean()