
import torch

# Local metrics

def procrustes(X, Y, scaling=True):
    """
    Reimplementation of MATLAB's `procrustes` function to Pytorch.
    """
    muX     = torch.mean(X, dim=1, keepdim=True) # GT shape N x J x C
    muY     = torch.mean(Y, dim=1, keepdim=True) # Pred shape N x J x C
    X0      = X - muX
    Y0      = Y - muY
    ssX     = torch.sum(torch.sum(X0**2., dim=-1, keepdim=True), dim=-2, keepdim=True)
    ssY     = torch.sum(torch.sum(Y0**2., dim=-1, keepdim=True), dim=-2, keepdim=True)

    # centred Frobenius norm
    normX   = torch.sqrt(ssX)
    normY   = torch.sqrt(ssY)

    # scale to equal (unit) norm
    X0      /= normX
    Y0      /= normY

    # optimum rotation matrix of Y
    A       = torch.bmm(X0.permute(0,2,1), Y0)
    U,s,Vt  = torch.linalg.svd(A,full_matrices=False)
    V       = Vt.permute(0,2,1)
    T       = torch.bmm(V, U.permute(0,2,1))

    V[:,:,-1] *= torch.sign(torch.linalg.det(T)).unsqueeze(1)
    s[:,-1]   *= torch.sign(torch.linalg.det(T))
    T       = torch.bmm(V, U.permute(0,2,1))
    traceTA = torch.sum(s, dim=-1, keepdim=True).unsqueeze(1)

    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*torch.bmm(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*torch.bmm(Y0, T) + muX
    c = muX - b*torch.bmm(muY, T)

    return d, Z, T, b, c

def mpjpe(predicted, target):
    """
    Mean Per-Joint Position Error (MPJPE), referred to as "Protocol #1".
    """
    assert predicted.shape == target.shape # N x T x J x C
    predicted       = predicted * 1e3
    target          = target * 1e3
    mask            = (target == 0).all(dim=(-2,-1))
    L2_dist         = torch.sqrt(((target-predicted)**2).sum(-1)).mean(-1) * (~mask)
    L2_dist         = L2_dist.sum(-1) / (~mask).sum(-1)
    return L2_dist

def mpjpe_pa(predicted, target):
    """
    Procrusted-Aligned Mean Per-Joint Position Error (MPJPE-PA), referred to as "Protocol #2".
    """
    assert predicted.shape == target.shape # N x T x J x C
    predicted       = predicted * 1e3
    target          = target * 1e3
    mask            = (target == 0).all(dim=(-2,-1))
    _, _, T, b, c   = procrustes(target[~mask], predicted[~mask], scaling=True)
    frame_pred      = (b * torch.bmm(predicted[~mask], T)) + c
    predicted_pa   = torch.zeros_like(target)
    predicted_pa[~mask] = frame_pred
    L2_dist         = torch.sqrt(((target-predicted_pa)**2).sum(-1)).mean(-1) * (~mask)
    L2_dist         = L2_dist.sum(-1) / (~mask).sum(-1)
    return L2_dist

def accel(p_ra17, pgt_ra17):
    assert p_ra17.shape == pgt_ra17.shape # N x T x J x C
    p_ra17      = p_ra17 * 1e3
    pgt_ra17    = pgt_ra17 * 1e3
    mask        = (pgt_ra17 == 0).all(dim=(-2,-1))
    accel_gt    = pgt_ra17[:,:-2,...] - 2 * pgt_ra17[:,1:-1,...] + pgt_ra17[:,2:,...]
    accel_pred  = p_ra17[:,:-2,...] - 2 * p_ra17[:,1:-1,...] + p_ra17[:,2:,...]
    L2_dist     = torch.norm(accel_pred - accel_gt, dim=-1)[~mask[:,2:]]
    return L2_dist.mean(-1)

def mpjpe_2d(predicted, target):
    """
    Mean Per-Joint Position Error (MPJPE) for 2D".
    """
    assert predicted.shape == target.shape # N x T x J x 2
    mask            = (target == 0).all(dim=(-2,-1))
    L2_dist         = (torch.sqrt(((target-predicted)**2).sum(-1)))[~mask]
    return L2_dist.mean(-1)

# Global metrics

def mpjpe_g(predicted, target):
    """
    Mean Per-Joint Position Error (MPJPE) without root alignment".
    """
    assert predicted.shape == target.shape # N x T x J x C
    predicted   = predicted * 1e3
    target      = target * 1e3
    mask        = (target == 0).all(dim=(-2,-1))
    L2_dist     = torch.sqrt(((target-predicted)**2).sum(-1)).mean(-1) * (~mask)
    L2_dist     = L2_dist.sum(-1) / (~mask).sum(-1)
    return L2_dist

def gre(predicted, target):
    """
    Mean Per-Joint Position Error (MPJPE) without root alignment".
    """
    assert predicted.shape == target.shape # N x T x C
    predicted   = predicted * 1e3
    target      = target * 1e3
    mask        = (target == 0).all(dim=(-1))
    L2_dist     = torch.sqrt(((target-predicted)**2).sum(-1)) * (~mask)
    L2_dist     = L2_dist.sum(-1) / (~mask).sum(-1)
    return L2_dist

def accel_g(p, pgt):
    assert p.shape == pgt.shape # N x T x J x C
    p           = p * 1e3
    pgt         = pgt * 1e3
    mask        = (pgt == 0).all(dim=(-1))
    accel_gt    = pgt[:,:-2,...] - 2 * pgt[:,1:-1,...] + pgt[:,2:,...]
    accel_pred  = p[:,:-2,...] - 2 * p[:,1:-1,...] + p[:,2:,...]
    L2_dist     = torch.norm(accel_pred - accel_gt, dim=-1)[~mask[:,2:]]
    return L2_dist

def foot_skate(seq_mesh, seq_p, threshold):
    Li      = [3428, 3461, 3462, 3456, 3449, 3448, 3443, 3438, 3359, 3360]
    Ri      = [6865, 6855, 6850, 6847, 6843, 6838, 6759, 6755, 6754, 6754]
    left_vert   = (seq_mesh)[:,:,Li]
    right_vert  = (seq_mesh)[:,:,Ri]
    contactL    = torch.all(left_vert[:,:,:,2] < threshold, dim=-1)
    contactR    = torch.all(right_vert[:,:,:,2] < threshold, dim=-1)
    left_moved  = torch.norm((seq_p[:,1:,3,:] - seq_p[:,:-1,3,:]), dim=-1)
    right_moved = torch.norm((seq_p[:,1:,6,:] - seq_p[:,:-1,6,:]), dim=-1)
    movedL_2cm  = left_moved >= 0.02
    movedR_2cm  = right_moved >= 0.02
    fkL         = 100 * (contactL[:,0:-1] * movedL_2cm).sum(-1) / 99 # [N]
    fkR         = 100 * (contactR[:,0:-1] * movedR_2cm).sum(-1) / 99 # [N]
    fk          = (fkL + fkR)
    return fk

def foot_skate_coco(seq_mesh, seq_p, threshold):
    N, T, _, _ = seq_mesh.shape
    Li      = [3428, 3461, 3462, 3456, 3449, 3448, 3443, 3438, 3359, 3360]
    Ri      = [6865, 6855, 6850, 6847, 6843, 6838, 6759, 6755, 6754, 6754]
    left_vert   = (seq_mesh)[:,:,Li]
    right_vert  = (seq_mesh)[:,:,Ri]
    contactL    = torch.all(left_vert[:,:,:,2] < threshold, dim=-1)
    contactR    = torch.all(right_vert[:,:,:,2] < threshold, dim=-1)
    left_moved  = torch.norm((seq_p[:,1:,15,:] - seq_p[:,:-1,15,:]), dim=-1)
    right_moved = torch.norm((seq_p[:,1:,16,:] - seq_p[:,:-1,16,:]), dim=-1)
    movedL_2cm  = left_moved >= 0.02
    movedR_2cm  = right_moved >= 0.02
    fkL         = 100 * (contactL[:,0:-1] * movedL_2cm).sum(-1) / (T-1) # [N]
    fkR         = 100 * (contactR[:,0:-1] * movedR_2cm).sum(-1) / (T-1) # [N]
    fk          = (fkL + fkR)
    return fk

def ground_penetration(seq_mesh, threshold):
    left_vert   = (seq_mesh)[:,:,leftFoot]
    right_vert  = (seq_mesh)[:,:,rightFoot]
    negL_vert   = torch.relu(torch.ones_like(left_vert[...,2])*threshold - left_vert[...,2]).mean(-1)
    negR_vert   = torch.relu(torch.ones_like(right_vert[...,2])*threshold - right_vert[...,2]).mean(-1)
    return 1000*(negL_vert + negR_vert)/2