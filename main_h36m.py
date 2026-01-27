import os
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import wandb
import common.utilities as utils
import common.data as data
import common.metric as metric
import common.loss as losses
import models.network_h36m as net
from common.utilities import print_log
from torch.utils.data import DataLoader
from alive_progress import alive_bar
from prettytable import PrettyTable
from pytorch3d import transforms
project_path = "./project"
h36m_17j = [0, 6,7,8, 1,2,3, 12, 13,14,15, 17,18,19, 25,26,27]

def main_train(args, init_net, ctrl_net, trainset, smpl_layer):
    loader_train    = DataLoader(dataset=trainset, batch_size=args.bsize, shuffle=True)
    params          = list(init_net.parameters()) + list(ctrl_net.parameters())
    optimizer       = torch.optim.AdamW(params, lr=args.lrate, weight_decay=0.02)
    scheduler       = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lstep, gamma=0.1)
    print_log('InitNet params: {:d}'.format(sum(p.numel() for p in init_net.parameters())))
    print_log('CtrlNet params: {:d}'.format(sum(p.numel() for p in ctrl_net.parameters())))
    print_log('Total epochs: {:d} \n'.format(args.epochs))
    for epoch in range(args.epochs):
        if epoch < args.warm:
            for _, param_group in enumerate(optimizer.param_groups): param_group['lr'] = 1e-4
        if epoch == args.warm:
            for _, param_group in enumerate(optimizer.param_groups): param_group['lr'] = args.lrate
        print('Epoch {:d}, lrate={:.2e}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        init_net.train()
        ctrl_net.train()
        tp1, tL, tLt, tLp, tL2d, tLts, tLps = [], [], [], [], [], [], []
        bar_length  = math.ceil(len(trainset)/args.bsize)
        with alive_bar(total=bar_length, title='Train:', length=20, bar='circles', dual_line=True) as bar:
            for _, batch in enumerate(loader_train):
                p3d_gt      = batch["p3d_gt"].to(device, dtype=torch.float64)   # N, T, 32, 3
                p2d_gt      = batch["p2d_gt"].to(device, dtype=torch.float64)   # N, T, 32, 2
                trans_in    = batch["trans"].to(device, dtype=torch.float64)    # N, T, 3
                betas_in    = batch["betas"].to(device, dtype=torch.float64)    # N, T, 10
                thetas_in   = batch["thetas"].to(device, dtype=torch.float64)   # N, T, 24, 3 - axis-angle
                ins_params  = batch["intrinsic"].to(device, dtype=torch.float64) # N, 4
                ins_mat     = utils.build_intrinsics(ins_params).unsqueeze(1)
                N, T, _, _  = p3d_gt.shape
                trans_out, linvel, quats_out, angvel, linerr, angerr = [], [], [], [], [], []
                quats_in    = utils.transform_to_quaternion(thetas_in) # don't standardize it
                mesh_in, j24_in = utils.SMPL_forward(smpl_layer, betas_in, quats_in) # N, T, ...
                [m0, v0, q0, w0], beta_fix = init_net(trans_in[:,:2], quats_in[:,:2], betas_in[:,0], thetas_in[:,0], j24_in[:,0])
                trans_out.append(m0)
                quats_out.append(q0)
                linvel.append(v0)
                angvel.append(w0)
                linerr.append(torch.zeros_like(v0))
                angerr.append(torch.zeros_like(w0))
                
                # Ground truth p2d, p3d with corresponding 17 joint format
                p2d_gt_17j  = p2d_gt[:,:,h36m_17j]
                p3d_gt_17j  = p3d_gt[:,:,h36m_17j]
                p3d_gt_17j_ra = p3d_gt_17j - p3d_gt_17j[...,:1,:]
                if epoch < args.warm:
                    for t in range(T-1):
                        mt, vt, qt, wt, emt, eqt = ctrl_net(
                            x_t = (trans_out[t], linvel[t], quats_out[t], angvel[t], linerr[t], angerr[t]),
                            x_i = (trans_in[:,t:t+2], quats_in[:,t:t+2]), t = t)
                        betas_t     = betas_in[:,0] + beta_fix
                        mesh_out, _ = utils.SMPL_forward(smpl_layer, betas_t, qt.unsqueeze(1))
                        J_regressor = ctrl_net.get_new_regressor()
                        _, p3d_17j_ra = utils.SMPL_regression(mesh_out, J_regressor)
                        p3d_17j     = mt[:,None,None,:] + p3d_17j_ra
                        p2d_17j_h   = (ins_mat @ p3d_17j.transpose(-2,-1)).transpose(-2,-1)
                        p2d_17j     = p2d_17j_h[:,:,:,:2] / p2d_17j_h[:,:,:,-1,None]
                        
                        trans_loss  = losses.trans_l1_loss(mt, p3d_gt_17j[:,t+1,0])
                        pose_loss   = losses.pose_l1_loss(p3d_17j_ra, p3d_gt_17j_ra[:,t+1,None])
                        proj_loss   = losses.proj_l1_loss(p2d_17j*1e-3, p2d_gt_17j[:,t+1,None]*1e-3) # normed pixels
                        loss        = 0.1*trans_loss + pose_loss + proj_loss
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(params, args.clip)
                        optimizer.step()

                        mt  = mt.clone().detach()
                        vt  = vt.clone().detach()
                        qt  = qt.clone().detach()
                        wt  = wt.clone().detach()
                        emt = emt.clone().detach()
                        eqt = eqt.clone().detach()
                        beta_fix = beta_fix.clone().detach()
                        
                        trans_out.append(mt)
                        linvel.append(vt)
                        quats_out.append(qt)
                        angvel.append(wt)
                        linerr.append(emt)
                        angerr.append(eqt)
                        bar.text = ('loss: {:.6f}'.format(loss))
                else:
                    for t in range(T-1):
                        mt, vt, qt, wt, emt, eqt = ctrl_net(
                            x_t = (trans_out[t], linvel[t], quats_out[t], angvel[t], linerr[t], angerr[t]),
                            x_i = (trans_in[:,t:t+2], quats_in[:,t:t+2]), t = t)
                        trans_out.append(mt)
                        linvel.append(vt)
                        quats_out.append(qt)
                        angvel.append(wt)
                        linerr.append(emt)
                        angerr.append(eqt)
                    trans_out   = torch.stack(trans_out, dim=1)
                    quats_out   = torch.stack(quats_out, dim=1)
                    betas_t     = betas_in[:,0,None].repeat(1,T,1) + beta_fix.unsqueeze(1)
                    mesh_out, _ = utils.SMPL_forward(smpl_layer, betas_t, quats_out)
                    J_regressor = ctrl_net.get_new_regressor()
                    _, p3d_17j_ra  = utils.SMPL_regression(mesh_out, J_regressor)
                    p3d_17j     = trans_out.unsqueeze(-2) + p3d_17j_ra
                    p2d_17j_h   = (ins_mat.repeat(1,T,1,1) @ p3d_17j.transpose(-2,-1)).transpose(-2,-1)
                    p2d_17j     = p2d_17j_h[:,:,:,:2] / p2d_17j_h[:,:,:,-1,None]
                    p1          = metric.mpjpe(p3d_17j_ra, p3d_gt_17j_ra).mean()
                    trans_loss  = losses.trans_l1_loss(trans_out, p3d_gt_17j[...,0,:])
                    pose_loss   = losses.pose_l1_loss(p3d_17j_ra, p3d_gt_17j_ra)
                    proj_loss   = losses.proj_l1_loss(p2d_17j*1e-3, p2d_gt_17j*1e-3) # normed pixels
                    smooth_t_loss = losses.smooth_trans_l1_loss(trans_out, p3d_gt_17j[...,0,:])
                    smooth_p_loss = losses.smooth_pose_l1_loss(p3d_17j_ra, p3d_gt_17j_ra)
                    betas_loss  = torch.abs(beta_fix).mean() # maintaining normal beta values
                    loss        = (0.1*trans_loss + pose_loss + proj_loss +
                                   0.1*smooth_t_loss + smooth_p_loss + 1e-2*betas_loss)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, args.clip)
                    optimizer.step()
                    
                    tL.append(loss.detach().cpu())
                    tp1.append(p1.detach().cpu())
                    tLt.append(trans_loss.detach().cpu())
                    tLp.append(pose_loss.detach().cpu())
                    tL2d.append(proj_loss.detach().cpu())
                    tLts.append(smooth_t_loss.detach().cpu())
                    tLps.append(smooth_p_loss.detach().cpu())
                    bar.text = ('Loss: {:.6f}'.format(loss))
                bar()
                
        if epoch > args.warm:
            if args.wandb: wandb.log({'epoch': epoch+1, 'MPJPE_train': np.mean(tp1)})
            print_log('MPJPE:  {:.2f}'.format(np.mean(tp1)))
            print_log('Loss:   {:.4f}'.format(np.mean(tL)))
            print_log('TLoss:  {:.4f}'.format(np.mean(tLt)))
            print_log('PLoss:  {:.4f}'.format(np.mean(tLp)))
            print_log('2DLoss: {:.4f}'.format(np.mean(tL2d)))
            print_log('STLoss: {:.4f}'.format(np.mean(tLts)))
            print_log('SPLoss: {:.4f}'.format(np.mean(tLps)))
        scheduler.step()
    
    init_net.save(save_path, name=f"InitNet_e{int(args.exact)}_s{int(args.second)}_{args.seed}.pth")
    ctrl_net.save(save_path, name=f"CtrlNet_e{int(args.exact)}_s{int(args.second)}_{args.seed}.pth")
    print('Training finished! \n')

def main_test(args, init_net, ctrl_net, testset, smpl_layer, results_path):
    loader_test = DataLoader(dataset=testset, batch_size=args.bsize, shuffle=False)
    init_net.load(save_path, name=f"InitNet_e{int(args.exact)}_s{int(args.second)}_{args.seed}.pth", device=device)
    ctrl_net.load(save_path, name=f"CtrlNet_e{int(args.exact)}_s{int(args.second)}_{args.seed}.pth", device=device)
    init_net.eval()
    ctrl_net.eval()
    crits = ['p1', 'p2', 'p3', 'acc', 'gp1', 'gre', 'gacc', 'fk']
    metrics_input, metrics_aqde = {}, {}
    for crit in crits:
        metrics_input[crit] = []
        metrics_aqde[crit] = []
    bar_length  = math.ceil(len(testset)/args.bsize)
    with alive_bar(total=bar_length, title='Testing:', length=20, bar='bubbles') as bar:
        with torch.no_grad():
            for bid, batch in enumerate(loader_test):
                p3d_gt      = batch["p3d_gt"].to(device, dtype=torch.float64)   # N, T, 32, 3
                p2d_gt      = batch["p2d_gt"].to(device, dtype=torch.float64)   # N, T, 32, 2
                trans_in    = batch["trans"].to(device, dtype=torch.float64)    # N, T, 3
                betas_in    = batch["betas"].to(device, dtype=torch.float64)    # N, T, 10
                thetas_in   = batch["thetas"].to(device, dtype=torch.float64)   # N, T, 24, 3 - axis-angle
                ins_params  = batch["intrinsic"].to(device, dtype=torch.float64) # N, 4
                ins_mat     = utils.build_intrinsics(ins_params).unsqueeze(1)
                N, T, _, _  = p3d_gt.shape
                
                trans_out, linvel, quats_out, angvel, linerr, angerr = [], [], [], [], [], []
                quats_in    = utils.transform_to_quaternion(thetas_in) # don't standardize it
                mesh_in, j24_in = utils.SMPL_forward(smpl_layer, betas_in, quats_in) # N, T, ...
                [m0, v0, q0, w0], beta_fix = init_net(trans_in[:,:2], quats_in[:,:2], betas_in[:,0], thetas_in[:,0], j24_in[:,0])
                trans_out.append(m0)
                quats_out.append(q0)
                linvel.append(v0)
                angvel.append(w0)
                linerr.append(torch.zeros_like(v0))
                angerr.append(torch.zeros_like(w0))
                for t in range(T-1):
                    mt, vt, qt, wt, emt, eqt = ctrl_net(
                        x_t = (trans_out[t], linvel[t], quats_out[t], angvel[t], linerr[t], angerr[t]),
                        x_i = (trans_in[:,t:t+2], quats_in[:,t:t+2]), t = t)
                    trans_out.append(mt)
                    linvel.append(vt)
                    quats_out.append(qt)
                    angvel.append(wt)
                    linerr.append(emt)
                    angerr.append(eqt)
                
                # Obtaining input results
                mesh_in_ra, p3d_in_ra = utils.SMPL_regression(mesh_in.double(), J_regressor_h36m)
                p3d_in      = trans_in.unsqueeze(-2) + p3d_in_ra
                p2d_in_h    = (ins_mat.repeat(1,T,1,1) @ p3d_in.transpose(-2,-1)).transpose(-2,-1)
                p2d_in      = p2d_in_h[:,:,:,:2] / p2d_in_h[:,:,:,-1,None]
                
                # Obtaining ground truths
                p2d_gt_17j  = p2d_gt[:,:,h36m_17j]
                p3d_gt_17j  = p3d_gt[:,:,h36m_17j,:]
                p3d_gt_17j_ra  = p3d_gt_17j - p3d_gt_17j[:,:,0:1]
                
                # Obtaining output results
                trans_out   = torch.stack(trans_out, dim=1)
                quats_out   = torch.stack(quats_out, dim=1)
                if args.input == 'hmr2': betas_t = betas_in
                else: betas_t = betas_in[:,0,None].repeat(1,T,1) + beta_fix.unsqueeze(1)
                mesh_out, j24_out = utils.SMPL_forward(smpl_layer, betas_t, quats_out)
                J_regressor = ctrl_net.get_new_regressor()
                mesh_out_ra, p3d_17j_ra  = utils.SMPL_regression(mesh_out, J_regressor)
                p3d_17j     = trans_out.unsqueeze(-2) + p3d_17j_ra
                p2d_17j_h   = (ins_mat.repeat(1,T,1,1) @ p3d_17j.transpose(-2,-1)).transpose(-2,-1)
                p2d_17j     = p2d_17j_h[:,:,:,:2] / p2d_17j_h[:,:,:,-1,None]
                
                # Computer foot skate
                extrinsic   = batch["extrinsic"].to(device, dtype=torch.float64)
                intrinsic   = batch["intrinsic"].to(device, dtype=torch.float64)
                cam_R       = transforms.euler_angles_to_matrix(extrinsic[:, :3],'XYZ').double()[:,None,None,:,:]
                cam_T       = extrinsic[:, None, None, None, 3:].double() * 1e-3
                mesh_in_cc  = trans_in.unsqueeze(-2) + mesh_in_ra
                mesh_out_cc = trans_out.unsqueeze(-2) + mesh_out_ra
                mesh_in_wc  = (mesh_in_cc.unsqueeze(-2)  @ cam_R - cam_T @ cam_R).squeeze(-2)
                mesh_out_wc = (mesh_out_cc.unsqueeze(-2) @ cam_R - cam_T @ cam_R).squeeze(-2)
                p3d_in_wc   = (p3d_in.unsqueeze(-2)  @ cam_R - cam_T @ cam_R).squeeze(-2)
                p3d_17j_wc  = (p3d_17j.unsqueeze(-2) @ cam_R - cam_T @ cam_R).squeeze(-2)
                if args.input == 'hmr2': ground = -0.02
                if args.input == 'trace': ground = 0.02
                
                # Relevant metrics
                metrics_input['p1'].append(metric.mpjpe(p3d_in_ra, p3d_gt_17j_ra).detach().cpu())
                metrics_input['p2'].append(metric.mpjpe_pa(p3d_in_ra, p3d_gt_17j_ra).detach().cpu())
                metrics_input['p3'].append(metric.mpjpe_2d(p2d_in, p2d_gt_17j).detach().cpu())
                metrics_input['acc'].append(metric.accel(p3d_in_ra, p3d_gt_17j_ra).detach().cpu())
                metrics_input['gp1'].append(metric.mpjpe_g(p3d_in, p3d_gt_17j).detach().cpu())
                metrics_input['gre'].append(metric.gre(trans_in, p3d_gt_17j[:,:,0]).detach().cpu())
                metrics_input['gacc'].append(metric.accel_g(p3d_in, p3d_gt_17j).detach().cpu())
                metrics_input['fk'].append(metric.foot_skate(mesh_in_wc, p3d_in_wc, ground))
                
                metrics_aqde['p1'].append(metric.mpjpe(p3d_17j_ra, p3d_gt_17j_ra).detach().cpu())
                metrics_aqde['p2'].append(metric.mpjpe_pa(p3d_17j_ra, p3d_gt_17j_ra).detach().cpu())
                metrics_aqde['p3'].append(metric.mpjpe_2d(p2d_17j, p2d_gt_17j).detach().cpu())
                metrics_aqde['acc'].append(metric.accel(p3d_17j_ra, p3d_gt_17j_ra).detach().cpu())
                metrics_aqde['gp1'].append(metric.mpjpe_g(p3d_17j, p3d_gt_17j).detach().cpu())
                metrics_aqde['gre'].append(metric.gre(trans_out, p3d_gt_17j[:,:,0]).detach().cpu())
                metrics_aqde['gacc'].append(metric.accel_g(p3d_17j, p3d_gt_17j).detach().cpu())
                metrics_aqde['fk'].append(metric.foot_skate(mesh_out_wc, p3d_17j_wc, ground))
                
                bar()
    
    tab_names, tab_input, tab_aqde = ['Method'], [args.input], ['quamo']
    for crit in crits:
        metrics_input[crit] = torch.cat(metrics_input[crit]).mean()
        metrics_aqde[crit]  = torch.cat(metrics_aqde[crit]).mean()
        tab_names.append(crit+"\u2193")
        tab_input.append('{:.1f}'.format(metrics_input[crit]))
        tab_aqde.append('{:.1f}'.format(metrics_aqde[crit]))
    
    print("\nAverage results:")
    table               = PrettyTable()
    table.field_names   = tab_names
    table.add_row(tab_input)
    table.add_row(tab_aqde)
    table.align["Type"] = "r"
    print(table)
    print()
    
    if args.wandb:
        wandb.log({'MPJPE_test': metrics_aqde['p1']})
        wandb.log({'P-MPJPE_test': metrics_aqde['p2']})
        wandb.log({'MPJPE-2D_test': metrics_aqde['p3']})
        wandb.log({'Accel_test': metrics_aqde['acc']})
        wandb.log({'G-MPJPE_test': metrics_aqde['gp1']})
        wandb.log({'GRE_test': metrics_aqde['gre']})
        wandb.log({'G-Accel_test': metrics_aqde['gacc']})
        wandb.log({'FK_test': metrics_aqde['fk']})
            
if __name__ == "__main__":
    args    = utils.get_parse().parse_args()
    if args.wandb:
        wandb.init(
            project = "QuaMo_"+args.dataset+"_"+args.exp,
            name = "Model",
            config = vars(args),
            dir = os.path.join(project_path, 'wandb')
            )
    utils.seed_everything(args.seed)
    torch.set_default_dtype(torch.float64)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using GPU: " + torch.cuda.get_device_name(device=device))
    
    smpls   = utils.load_smpl_models(project_path, device) # this is a dict
    J_regressor_h36m = torch.from_numpy(
        np.load(os.path.join(project_path, "smpl_models", "J_regressor_h36m.npy"))
    ).to(device)
    trainset, testset = data.create_dataset_h36m(args, project_path)
    init_net    = net.InitNet(hid_dim=128).to(device, dtype=torch.float64)
    ctrl_net    = net.CtrlNet(args=args, hid_dim=512, J_reg=J_regressor_h36m).to(device, dtype=torch.float64)
    save_path   = os.path.join(project_path, 'trained_models', args.dataset, args.exp, args.input)
    if args.train: main_train(args, init_net, ctrl_net, trainset, smpls["neutral"])
    results_path = os.path.join(project_path, "saved_results", "h36m_results_"+args.input+".pkl")
    main_test(args, init_net, ctrl_net, testset, smpls["neutral"], results_path)
    print("-- Finished --")