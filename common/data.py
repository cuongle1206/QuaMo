
import os
import numpy as np
import h5py, json
import pickle
import torch
from tqdm import tqdm
from time import time
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import median_filter

import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

def pad_to_nearest_multiple(arr, multiple=100, pad_value=0):
    n = arr.shape[0]
    target = ((n + multiple - 1) // multiple) * multiple
    pad_width = target - n
    if len(arr.shape) == 3: pad_config = ((0, pad_width), (0, 0), (0, 0))
    if len(arr.shape) == 2: pad_config = ((0, pad_width), (0, 0))
    return np.pad(arr, pad_config, mode='constant', constant_values=pad_value)


def create_dataset_h36m(args, project_path):
    dataset_path = "./datasets/h36m"
    if args.exp == 'ablation': save_path = os.path.join(project_path, "database", "h36m", f"{args.exp}_{args.input}")
    else: save_path = os.path.join(project_path, "database", "h36m", args.input)
    os.makedirs(save_path, exist_ok=True)
    
    if len(os.listdir(save_path))!=0:
        with open(os.path.join(save_path, "trainset.pkl"), 'rb') as f:
            trainset = pickle.load(f)
        with open(os.path.join(save_path, "testset.pkl"), 'rb') as f:
            testset = pickle.load(f)
    else:
        cameras     = ["54138969", "55011271", "58860488", "60457274"] # 0, 1, 2, 3
        subjects    = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]
        train_sj, test_sj = ["S1", "S5", "S6", "S7", "S8"], ["S9", "S11"]
        actions1    = [ "Directions-1", "Discussion-1", "Greeting-1", "Posing-1", "Purchases-1",
                       "TakingPhoto-1", "Waiting-1", "Walking-1", "WalkingDog-1", "WalkingTogether-1"]
        actions2    = [ "Directions-2", "Discussion-2", "Greeting-2", "Posing-2", "Purchases-2", 
                       "TakingPhoto-2", "Waiting-2", "Walking-2", "WalkingDog-2", "WalkingTogether-2"]
        actions     = [item for pair in zip(actions1, actions2) for item in pair]
        cam_params  = np.load(dataset_path + "/camera_h36m.npy")
        
        trainset, testset = [], []
        for sid, subject in (enumerate(subjects)):
            for _, action in (enumerate(actions)):
                if subject in train_sj:
                    if args.exp == "ablation": views = [3]
                    else: views = list(range(4))
                if subject in test_sj:
                    if args.input == 'trace': views = [3]
                    if args.input == 'hmr2': views = list(range(4))
                for _, view in (enumerate(views)):
                    print(("Processing h36m " + subject + "," + cameras[view] + "," + action))
                    imgs_folder = os.path.join(dataset_path, "processed", subject, action, "imageSequence", cameras[view])
                    seqlen = len(os.listdir(imgs_folder))
                    start = time()
                    
                    if args.input == "trace":
                        data_path = os.path.join(dataset_path, "TRACE_results", subject, action, f"{cameras[view]}.npz")
                        data    = np.load(data_path, allow_pickle=True)['outputs'][()]
                        trans   = np.asarray(data.get('cam_trans'))       # m
                        betas   = np.asarray(data.get('smpl_betas'))      # betas
                        thetas  = np.asarray(data.get('smpl_thetas')).reshape(-1, 24, 3) # in axis-angle
                        
                    if args.input == "hmr2":
                        data_path = os.path.join(dataset_path, "HMR2_results", subject, action, f"{cameras[view]}.npz")
                        data    = np.load(data_path, allow_pickle=True)
                        trans   = data['trans'] # in Meters
                        orien   = data['orien'] # in rotation matrix
                        betas   = data['betas'] # betas
                        poses   = data['poses'] # in rotation matrix
                        thetas  = torch.cat((torch.Tensor(orien), torch.Tensor(poses)), dim=1)
                        thetas  = transforms.matrix_to_axis_angle(thetas).numpy()
                    
                    annot   = h5py.File(os.path.join(dataset_path, "processed", subject, action, "annot.h5"), 'r')
                    p3d_gt  = np.array(annot.get('pose')['3d-univ'])[(seqlen*view):(seqlen*view+seqlen), ...]*1e-3 # mm->m
                    p2d_gt  = np.array(annot.get('pose')['2d'])[(seqlen*view):(seqlen*view+seqlen), ...] # mm
                    
                    # first frame is globally aligned at GT
                    offset  = p3d_gt[0,0,:] - trans[0,:]
                    trans   = trans + offset
                    
                    # down-sample to 25Hz
                    p3d_gt  = pad_to_nearest_multiple(p3d_gt[::2], args.wsize)
                    p2d_gt  = pad_to_nearest_multiple(p2d_gt[::2], args.wsize)
                    if args.input == "trace":
                        trans   = pad_to_nearest_multiple(trans[::2], args.wsize)
                        betas   = pad_to_nearest_multiple(betas[::2], args.wsize)
                        thetas  = pad_to_nearest_multiple(thetas[::2], args.wsize)
                    if args.input == "hmr2": # HMR2 is already downsampled
                        trans   = pad_to_nearest_multiple(trans, args.wsize)
                        betas   = pad_to_nearest_multiple(betas, args.wsize)
                        thetas  = pad_to_nearest_multiple(thetas, args.wsize)
                    seqlen_ds = p3d_gt.shape[0]
                    
                    cam_param = np.array(cam_params[sid][view])
                    
                    for k in range(int(seqlen_ds/args.wsize)):
                        sample              = {}
                        start_frame         = args.wsize*k
                        stop_frame          = args.wsize*k+args.wsize
                        sample["start"]     = start_frame
                        sample["stop"]      = stop_frame
                        sample["subject"]   = subject
                        sample["action"]    = action
                        sample["camera"]    = cameras[view]
                        sample["extrinsic"] = cam_param[:6] # [R,T], euler XYZ and T
                        sample["intrinsic"] = cam_param[6:10] # [fx,fy,cx,cy]
                        sample["p3d_gt"]    = torch.from_numpy(p3d_gt[start_frame:stop_frame])
                        sample["p2d_gt"]    = torch.from_numpy(p2d_gt[start_frame:stop_frame])
                        sample["trans"]     = torch.from_numpy(trans[start_frame:stop_frame])
                        sample["betas"]     = torch.from_numpy(betas[start_frame:stop_frame])
                        sample["thetas"]    = torch.from_numpy(thetas[start_frame:stop_frame])
                        if subject in train_sj: trainset.append(sample)
                        if subject in test_sj: testset.append(sample)

                    # print(f"Elapsed: {(time()-start):.2f}, len: {seqlen_ds}")
        with open(os.path.join(save_path, "trainset.pkl"), 'wb') as f:
            pickle.dump(trainset, f)
        with open(os.path.join(save_path, "testset.pkl"), 'wb') as f:
            pickle.dump(testset, f)
        
    print('Total training/testing samples: {:d}/{:d} \n'.format(len(trainset),len(testset)))
    return trainset, testset
    
    
def create_dataset_fit3d(args, project_path):
    dataset_path = "./datasets/fit3d"
    if args.exp == 'ablation': save_path = os.path.join(project_path, "database", "fit3d", f"{args.exp}_{args.input}")
    else: save_path = os.path.join(project_path, "database", "fit3d", args.input)
    os.makedirs(save_path, exist_ok=True)
    
    if len(os.listdir(save_path))!=0:
        with open(os.path.join(save_path, "trainset.pkl"), 'rb') as f:
            trainset = pickle.load(f)
        with open(os.path.join(save_path, "testset.pkl"), 'rb') as f:
            testset = pickle.load(f)
    else:
        subjects    = ["s03", "s04", "s05", "s07", "s08", "s10", "s09", "s11"]
        train_sj, test_sj = ["s03", "s04", "s05", "s07", "s08", "s10"], ["s09", "s11"]
        views       = ["60457274"]
        actions     = ["band_pull_apart", "barbell_dead_row", "barbell_row", "barbell_shrug",
                       "clean_and_press", "deadlift", "drag_curl", "dumbbell_biceps_curls", 
                       "dumbbell_curl_trifecta" ,"dumbbell_hammer_curls", "dumbbell_high_pulls", 
                       "dumbbell_overhead_shoulder_press", "dumbbell_reverse_lunge", "dumbbell_scaptions", 
                       "neutral_overhead_shoulder_press", "one_arm_row", "overhead_extension_thruster", 
                       "overhead_trap_raises", "side_lateral_raise", "squat", "standing_ab_twists", 
                       "w_raise", "walk_the_box", "warmup_2", "warmup_3", "warmup_4", "warmup_5",
                       "warmup_6", "warmup_7", "warmup_8", "warmup_9", "warmup_10", "warmup_11", 
                       "warmup_12", "warmup_13", "warmup_14", "warmup_15", "warmup_16", "warmup_17", "warmup_18", "warmup_19"]
        
        trainset, testset = [], []
        for sid, subject in (enumerate(subjects)):
            for _, action in (enumerate(actions)):
                for _, view in (enumerate(views)):
                    print(("Processing fit3d " + subject + "," + view + "," + action))
                    start = time()
                    
                    if args.input == "trace":
                        data_path = os.path.join(dataset_path, "TRACE_results", subject, view, action, f"{action}.mp4.npz")
                        data    = np.load(data_path, allow_pickle=True)['outputs'][()]
                        trans   = np.asarray(data.get('cam_trans'))       # m
                        betas   = np.asarray(data.get('smpl_betas'))      # betas
                        thetas  = np.asarray(data.get('smpl_thetas')).reshape(-1, 24, 3) # in axis-angle
                        
                    if args.input == "hmr2":
                        data_path = os.path.join(dataset_path, "HMR2_results", subject, action, f"{action}.npz")
                        data    = np.load(data_path, allow_pickle=True)
                        trans   = data['trans'] # in Meters
                        orien   = data['orien'] # in rotation matrix
                        betas   = data['betas'] # betas
                        poses   = data['poses'] # in rotation matrix
                        thetas  = torch.cat((torch.Tensor(orien), torch.Tensor(poses)), dim=1)
                        thetas  = transforms.matrix_to_axis_angle(thetas).numpy()
                    
                    j3d_path = os.path.join(dataset_path, 'train', subject, 'joints3d_25', action+".json")
                    with open(j3d_path) as json_file:
                        j3d_data = json.load(json_file)
                    ang_path = os.path.join(dataset_path, 'train', subject, 'smplx', action+".json")
                    with open(ang_path) as json_file:
                        ang_data = json.load(json_file)
                    param_path = os.path.join(dataset_path, 'train', subject, 'camera_parameters', view, action+".json")
                    with open(param_path) as json_file:
                        cam_param = json.load(json_file)
                    
                    p3d_gt_wc = np.array(j3d_data['joints3d_25'])
                    q_gt    = np.array(ang_data['transl'])
                    q_gt    = np.array(ang_data['global_orient'])
                    q_gt    = np.array(ang_data['body_pose'])
                    
                    # from world coord to camera coord
                    R       = np.array(cam_param['extrinsics']['R'])
                    T       = np.array(cam_param['extrinsics']['T'])
                    p3d_gt  = p3d_gt_wc @ R.T + T
                    
                    # first frame is globally aligned at GT
                    offset  = p3d_gt[0,0,:] - trans[0,:]
                    trans   = trans + offset
                    
                    # down-sample to 25Hz
                    p3d_gt  = pad_to_nearest_multiple(p3d_gt[::2], args.wsize)
                    if args.input == "trace":
                        trans   = pad_to_nearest_multiple(trans[::2], args.wsize)
                        betas   = pad_to_nearest_multiple(betas[::2], args.wsize)
                        thetas  = pad_to_nearest_multiple(thetas[::2], args.wsize)
                    if args.input == "hmr2": # HMR2 is already downsampled
                        trans   = pad_to_nearest_multiple(trans, args.wsize)
                        betas   = pad_to_nearest_multiple(betas, args.wsize)
                        thetas  = pad_to_nearest_multiple(thetas, args.wsize)
                    seqlen_ds = p3d_gt.shape[0]
                    
                    for k in range(int(seqlen_ds/args.wsize)):
                        sample              = {}
                        start_frame         = args.wsize*k
                        stop_frame          = args.wsize*k+args.wsize
                        sample["start"]     = start_frame
                        sample["stop"]      = stop_frame
                        sample["subject"]   = subject
                        sample["action"]    = action
                        sample["camera"]    = view
                        sample["extrinsic"] = cam_param['extrinsics'] # {'R', 'T'}
                        sample["intrinsic"] = cam_param['intrinsics_wo_distortion'] # {'c', 'f'}
                        sample["p3d_gt"]    = torch.from_numpy(p3d_gt[start_frame:stop_frame])
                        sample["trans"]     = torch.from_numpy(trans[start_frame:stop_frame])
                        sample["betas"]     = torch.from_numpy(betas[start_frame:stop_frame])
                        sample["thetas"]    = torch.from_numpy(thetas[start_frame:stop_frame])
                        if subject in train_sj: trainset.append(sample)
                        if subject in test_sj: testset.append(sample)

                    # print(f"Elapsed: {(time()-start):.2f}, len: {seqlen_ds}")
        with open(os.path.join(save_path, "trainset.pkl"), 'wb') as f:
            pickle.dump(trainset, f)
        with open(os.path.join(save_path, "testset.pkl"), 'wb') as f:
            pickle.dump(testset, f)
        
    print('Total training/testing samples: {:d}/{:d} \n'.format(len(trainset),len(testset)))
    return trainset, testset


def create_dataset_sport(args, project_path):
    dataset_path = "./datasets/SportsPose"
    if args.exp == 'ablation': save_path = os.path.join(project_path, "database", "sport", f"{args.exp}_{args.input}")
    else: save_path = os.path.join(project_path, "database", "sport", args.input)
    os.makedirs(save_path, exist_ok=True)
    
    if len(os.listdir(save_path))!=0:
        with open(os.path.join(save_path, "trainset.pkl"), 'rb') as f:
            trainset = pickle.load(f)
        with open(os.path.join(save_path, "testset.pkl"), 'rb') as f:
            testset = pickle.load(f)
    else:
        subjects    = ["S02", "S03", "S05", "S06", "S07", "S08", "S09", "S12", "S13", "S14"]
        train_sj, test_sj = ["S02", "S03", "S05", "S06", "S07", "S08", "S09"], ["S12", "S13", "S14"]
        actions     = ["jump0000", "jump0001", "jump0002", "jump0003", "jump0004",
                        "throw_baseball0005", "throw_baseball0006", "throw_baseball0007", "throw_baseball0008", "throw_baseball0009",
                        "soccer0010", "soccer0011", "soccer0012", "soccer0013", "soccer0014",
                        "volley0015", "volley0016", "volley0017", "volley0018", "volley0019",
                        "tennis0020", "tennis0021", "tennis0022", "tennis0023", "tennis0024"]
        
        ang         = np.pi/2
        Rz          = np.array([
            [np.cos(ang), -np.sin(ang), 0, 0],
            [np.sin(ang), np.cos(ang), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
            ])
                
        trainset, testset = [], []
        for sid, subject in (enumerate(subjects)):
            for aid, action in (enumerate(actions)):
                print(("Processing sport " + subject + "," + action))
                start = time()
                
                if args.input == "trace":
                    data_path = os.path.join(dataset_path, "TRACE_results", "indoors", subject, action, 
                                                f"CAM3_rotated_video_{aid}.avi.npz")
                    data    = np.load(data_path, allow_pickle=True)['outputs'][()]
                    trans   = np.asarray(data.get('cam_trans'))       # m
                    betas   = np.asarray(data.get('smpl_betas'))      # betas
                    thetas  = np.asarray(data.get('smpl_thetas')).reshape(-1, 24, 3) # in axis-angle
                    
                if args.input == "hmr2":
                    data_path = os.path.join(dataset_path, "HMR2_results", subject, action, f"{action}.npz")
                    data    = np.load(data_path, allow_pickle=True)
                    trans   = data['trans'] # in Meters
                    orien   = data['orien'] # in rotation matrix
                    betas   = data['betas'] # betas
                    poses   = data['poses'] # in rotation matrix
                    thetas  = torch.cat((torch.Tensor(orien), torch.Tensor(poses)), dim=1)
                    thetas  = transforms.matrix_to_axis_angle(thetas).numpy()
                
                p3d_gt_wc   = np.load(os.path.join(dataset_path, 'data', 'indoors', subject, action+".npy"))
                param_path  = os.path.join(dataset_path, 'data', 'indoors', subject, "calib.pkl")
                with open(param_path, 'rb') as pkl_file:
                    cam_param = pickle.load(pkl_file)
                    cam_param = cam_param['calibration'][3]
                
                # from world coord to camera coord
                p3d_gt_wc[...,0] = -p3d_gt_wc[...,0]
                p3d_gt_wc[...,2] = -p3d_gt_wc[...,2]
                R       = np.array(cam_param['R'])
                T       = np.array(cam_param['T'])
                p3d_gt  = p3d_gt_wc #@ R.T + T
                root    = (p3d_gt[:,11,:] + p3d_gt[:,12,:])/2
                
                # c           = cam_param['c']
                # f           = cam_param['f']
                # I       = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0,0,1]])
                # ang     = np.pi/2
                # Rz      = np.array([[np.cos(ang), -np.sin(ang), 0],
                #                     [np.sin(ang), np.cos(ang), 0],
                #                     [0,0,1]])
                # fig     = plt.figure()
                # ax1     = fig.add_subplot(121)
                # ax2     = fig.add_subplot(122, projection='3d')
                # parent      = [1,1,2,2,3,4,5,  6,12,14, 7,13,15, 6,8,  7,9,  6,12]
                # child       = [2,3,3,4,5,6,7, 12,14,16,13,15,17, 8,10, 9,11, 7,13]
                # print(p3d_gt.shape)
                # p3d_gt_f    = p3d_gt[50]
                # ax2.set_box_aspect((1, 1, 1))
                # ax2.scatter(p3d_gt_f[:,0], p3d_gt_f[:,1], p3d_gt_f[:,2])
                # for k in range(len(parent)):
                #     ax2.plot([p3d_gt_f[parent[k]-1,0], p3d_gt_f[child[k]-1,0]], 
                #             [p3d_gt_f[parent[k]-1,1], p3d_gt_f[child[k]-1,1]], 
                #             [p3d_gt_f[parent[k]-1,2], p3d_gt_f[child[k]-1,2]],
                #             color='lime', linestyle='-', linewidth=3)
                # base_path       = "./datasets/SportsPose/extracted"
                # real_index      = 50
                # img_path        = os.path.join(base_path, subject, f"CAM3_video_12", f"frame_{real_index:06d}.jpg")
                # img             = plt.imread(img_path)
                # # img = ndimage.rotate(img, 90, reshape=True)
                # ax1.imshow(img)
                # p2d         = ((Rz@I) @ p3d_gt_f.T).T
                # print(p2d)
                # p2d_17j     = p2d[:,:2] / p2d[:,-1,None]
                # p2d_17j[:,0] += 1216
                # print(p2d_17j.shape)
                # for k in range(len(parent)):
                #     ax1.plot([p2d_17j[parent[k]-1,0], p2d_17j[child[k]-1,0]], 
                #             [p2d_17j[parent[k]-1,1], p2d_17j[child[k]-1,1]], 
                #             color='lime', linestyle='-', linewidth=3)
                # plt.show()
                
                cam_G               = np.eye(4) # Transformation matrix
                cam_G[:3,:3], cam_G[:3,3] = R, T
                cam_G_inv           = np.linalg.inv(cam_G)
                
                # Transform TRACE inputs from camera space to world space
                root_rmat_cc_obj    = Rot.from_rotvec(thetas[:,0,:])
                root_rmat_cc        = root_rmat_cc_obj.as_matrix()
                root_trans_wc       = np.zeros((trans.shape[0], 3))
                root_axis_wc        = np.zeros((trans.shape[0], 3))
                for f in range(trans.shape[0]):
                    root_trans_homog    = np.append(trans[f,:], 1.0)
                    root_trans_wc[f,:]  = (cam_G_inv @ -np.linalg.inv(Rz) @ root_trans_homog)[:3]
                    root_rmat_wc        = Rot.from_matrix(R.T @ Rz[:3,:3] @ root_rmat_cc[f,...])
                    root_axis_wc[f,:]   = root_rmat_wc.as_rotvec()
                
                trans = root_trans_wc
                thetas[:,0,:] = root_axis_wc
                
                vert_offset   = root[0,2] - trans[0,2]
                trans[:,:2]   = trans[:,:2] - trans[0,:2]
                trans[:,2]    = trans[:,2] + vert_offset
                p3d_gt[...,:2]  = p3d_gt[...,:2] - root[0,None,:2]
                
                # down-sample to 25Hz
                p3d_gt  = pad_to_nearest_multiple(p3d_gt[::2], args.wsize)
                if args.input == "trace":
                    trans   = pad_to_nearest_multiple(trans[::2], args.wsize)
                    betas   = pad_to_nearest_multiple(betas[::2], args.wsize)
                    thetas  = pad_to_nearest_multiple(thetas[::2], args.wsize)
                if args.input == "hmr2": # HMR2 is already downsampled
                    trans   = pad_to_nearest_multiple(trans, args.wsize)
                    betas   = pad_to_nearest_multiple(betas, args.wsize)
                    thetas  = pad_to_nearest_multiple(thetas, args.wsize)
                seqlen_ds = p3d_gt.shape[0]
                
                for k in range(int(seqlen_ds/args.wsize)):
                    sample              = {}
                    start_frame         = args.wsize*k
                    stop_frame          = args.wsize*k+args.wsize
                    sample["start"]     = start_frame
                    sample["stop"]      = stop_frame
                    sample["offset"]    = root[0,None,:2]
                    sample["subject"]   = subject
                    sample["action"]    = action
                    sample["camera"]    = "uniform"
                    sample["paramerters"] = cam_param
                    sample["p3d_gt"]    = torch.from_numpy(p3d_gt[start_frame:stop_frame])
                    sample["trans"]     = torch.from_numpy(trans[start_frame:stop_frame])
                    sample["betas"]     = torch.from_numpy(betas[start_frame:stop_frame])
                    sample["thetas"]    = torch.from_numpy(thetas[start_frame:stop_frame])
                    if subject in train_sj: trainset.append(sample)
                    if subject in test_sj: testset.append(sample)

        with open(os.path.join(save_path, "trainset.pkl"), 'wb') as f:
            pickle.dump(trainset, f)
        with open(os.path.join(save_path, "testset.pkl"), 'wb') as f:
            pickle.dump(testset, f)
        
    print('Total training/testing samples: {:d}/{:d} \n'.format(len(trainset),len(testset)))
    return trainset, testset


def create_dataset_aist(args, project_path):
    dataset_path = "./datasets/aist"
    if args.exp == 'ablation': save_path = os.path.join(project_path, "database", "aist", f"{args.exp}_{args.input}")
    else: save_path = os.path.join(project_path, "database", "aist", args.input)
    os.makedirs(save_path, exist_ok=True)
    
    if len(os.listdir(save_path))!=0:
        with open(os.path.join(save_path, "trainset.pkl"), 'rb') as f:
            trainset = pickle.load(f)
        with open(os.path.join(save_path, "testset.pkl"), 'rb') as f:
            testset = pickle.load(f)
    else:
        train_act = [
            "gHO_sFM_c01_d19_mHO3_ch04", # train
            "gKR_sFM_c01_d28_mKR3_ch04", 
            "gLH_sFM_c01_d16_mLH3_ch04",
            "gMH_sBM_c02_d24_mMH3_ch03", 
            "gMH_sBM_c03_d24_mMH3_ch06", 
            "gMH_sBM_c04_d24_mMH3_ch01",
            "gMH_sBM_c05_d24_mMH3_ch07", 
            "gMH_sBM_c06_d24_mMH3_ch05", 
            "gMH_sFM_c07_d24_mMH3_ch18", 
            "gMH_sBM_c08_d24_mMH3_ch04",
            "gMH_sBM_c09_d24_mMH3_ch09", 
        ]
        test_act = [
            "gBR_sBM_c06_d06_mBR4_ch06", # test
            "gBR_sBM_c07_d06_mBR4_ch02", 
            "gBR_sBM_c08_d05_mBR1_ch01",
            "gBR_sFM_c03_d04_mBR0_ch01", 
            "gJB_sBM_c02_d09_mJB3_ch10", 
            "gKR_sBM_c09_d30_mKR5_ch05",
            "gLH_sBM_c04_d18_mLH5_ch07", 
            "gLH_sBM_c07_d18_mLH4_ch03", 
            "gLH_sBM_c09_d17_mLH1_ch02",
            "gLH_sFM_c03_d18_mLH0_ch15", 
            "gLO_sBM_c05_d14_mLO4_ch07", 
            "gLO_sBM_c07_d15_mLO4_ch09",
            "gLO_sFM_c02_d15_mLO4_ch21", 
            "gMH_sBM_c01_d24_mMH3_ch02",
            "gMH_sBM_c05_d24_mMH4_ch07"
        ]
        actions = train_act + test_act

        labels  = [
            "gHO_sFM_cAll_d19_mHO3_ch04", 
            "gKR_sFM_cAll_d28_mKR3_ch04", 
            "gLH_sFM_cAll_d16_mLH3_ch04",
            "gMH_sBM_cAll_d24_mMH3_ch03", 
            "gMH_sBM_cAll_d24_mMH3_ch06", 
            "gMH_sBM_cAll_d24_mMH3_ch01",
            "gMH_sBM_cAll_d24_mMH3_ch07", 
            "gMH_sBM_cAll_d24_mMH3_ch05", 
            "gMH_sFM_cAll_d24_mMH3_ch18",
            "gMH_sBM_cAll_d24_mMH3_ch04", 
            "gMH_sBM_cAll_d24_mMH3_ch09",
            "gBR_sBM_cAll_d06_mBR4_ch06", # setting 1
            "gBR_sBM_cAll_d06_mBR4_ch02", # setting 1
            "gBR_sBM_cAll_d05_mBR1_ch01", # setting 1
            "gBR_sFM_cAll_d04_mBR0_ch01", # setting 1
            "gJB_sBM_cAll_d09_mJB3_ch10", # setting 3
            "gKR_sBM_cAll_d30_mKR5_ch05", # setting 5
            "gLH_sBM_cAll_d18_mLH5_ch07", # setting 6
            "gLH_sBM_cAll_d18_mLH4_ch03", # setting 6
            "gLH_sBM_cAll_d17_mLH1_ch02", # setting 6
            "gLH_sFM_cAll_d18_mLH0_ch15", # setting 6
            "gLO_sBM_cAll_d14_mLO4_ch07", # setting 7_1
            "gLO_sBM_cAll_d15_mLO4_ch09", # setting 7_1
            "gLO_sFM_cAll_d15_mLO4_ch21", # setting 7_2
            "gMH_sBM_cAll_d24_mMH3_ch02", # setting 8_1
            "gMH_sBM_cAll_d24_mMH4_ch07"
        ]
        
        settings = [
            "setting2", # train
            "setting5",
            "setting6",
            "setting8_1",
            "setting8_1",
            "setting8_1",
            "setting8_1",
            "setting8_1",
            "setting8_1",
            "setting8_1",
            "setting8_1",
            "setting1", # test
            "setting1", 
            "setting1", 
            "setting1",
            "setting3", 
            "setting5",
            "setting6", 
            "setting6", 
            "setting6", 
            "setting6",
            "setting7_1",
            "setting7_1", 
            "setting7_2",
            "setting7_2", 
            "setting8_1", 
            "setting8_1"
        ]
        
        camera_ids =  [1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                       6, 7, 8, 3, 2, 9, 4, 7, 9, 3, 5, 7, 2, 1, 5]
                
        trainset, testset = [], []
        for aid, action in (enumerate(actions)):
            print(("Processing AISTpp, " + action))
            start = time()
            
            if args.input == "trace":
                data_path = os.path.join(dataset_path, "subset", "TRACE_results", action, f"{action}.mp4.npz")
                data    = np.load(data_path, allow_pickle=True)['outputs'][()]
                trans   = np.asarray(data.get('cam_trans'))       # m
                betas   = np.asarray(data.get('smpl_betas'))      # betas
                thetas  = np.asarray(data.get('smpl_thetas')).reshape(-1, 24, 3) # in axis-angle
                
            if args.input == "hmr2":
                if action in train_act:
                    data_path = os.path.join(dataset_path, "train", "HMR2_results", f"{action}.npz")
                else:
                    data_path = os.path.join(dataset_path, "subset", "HMR2_results", f"{action}.npz")
                data    = np.load(data_path, allow_pickle=True)
                trans   = data['trans'] # in Meters
                orien   = data['orien'] # in rotation matrix
                betas   = data['betas'] # betas
                poses   = data['poses'] # in rotation matrix
                thetas  = torch.cat((torch.Tensor(orien), torch.Tensor(poses)), dim=1)
                thetas  = transforms.matrix_to_axis_angle(thetas).numpy()
            
            if action in train_act:
                label_path          = os.path.join(dataset_path, "train", "labels", f"{labels[aid]}.pkl")
                setting_path        = os.path.join(dataset_path, "train", "cameras", f"{settings[aid]}.json")
            else:
                label_path          = os.path.join(dataset_path, "subset", "labels", f"{labels[aid]}.pkl")
                setting_path        = os.path.join(dataset_path, "subset", "cameras", f"{settings[aid]}.json")
            with open(label_path, 'rb') as pkl_file:
                j3d_data = pickle.load(pkl_file)
            p3d_gt_wc = np.array(j3d_data['keypoints3d'])*1e-2
            with open(setting_path) as json_file:
                setting = json.load(json_file)
            
            ang = np.pi/2
            Rx  = np.array([[1, 0, 0],
                            [0, np.cos(ang), -np.sin(ang)],
                            [0, np.sin(ang), np.cos(ang)]])
            
            camera = setting[camera_ids[aid]-1]
            
            # from world coord to camera coord
            R       = Rot.from_rotvec(np.array(camera['rotation'])).as_matrix()
            T       = np.array(camera['translation'])*1e-2
            p3d_gt  = median_filter(p3d_gt_wc @ Rx.T, size=(5, 1, 1), mode='nearest')
            
            
            cam_G               = np.eye(4) # Transformation matrix
            cam_G[:3,:3], cam_G[:3,3] = R, T
            cam_G_inv           = np.linalg.inv(cam_G)
            
            # Transform TRACE inputs from camera space to world space
            root_rmat_cc_obj    = Rot.from_rotvec(thetas[:,0,:])
            root_rmat_cc        = root_rmat_cc_obj.as_matrix()
            root_trans_wc       = np.zeros((trans.shape[0], 3))
            root_axis_wc        = np.zeros((trans.shape[0], 3))
            for f in range(trans.shape[0]):
                root_trans_homog    = np.append(trans[f,:], 1.0)
                root_trans_wc[f,:]  = Rx @ (cam_G_inv @ root_trans_homog)[:3]
                root_rmat_wc        = Rot.from_matrix(Rx @ R.T @ root_rmat_cc[f,...])
                root_axis_wc[f,:]   = root_rmat_wc.as_rotvec()
            
            trans = root_trans_wc
            thetas[:,0,:] = root_axis_wc
            
            root    = (p3d_gt[:,11,:] + p3d_gt[:,12,:])/2
            vert_offset   = root[0,2] - trans[0,2]
            trans[:,:2]   = trans[:,:2] - trans[0,:2]
            trans[:,2]    = trans[:,2] + vert_offset
            p3d_gt[...,:2]  = p3d_gt[...,:2] - root[0,None,:2]
            p3d_gt[...,2] -= 0.7
            trans[...,2]  -= 0.7
            
            sample              = {}
            start_frame         = 0#args.wsize*k
            stop_frame          = 120#args.wsize*k+args.wsize
            sample["start"]     = start_frame
            sample["stop"]      = stop_frame
            sample["offset"]    = root[0,None,:2]
            sample["subject"]   = "dance"
            sample["action"]    = action
            sample["camera"]    = camera_ids[aid]-1
            sample["parameters"] = camera
            # sample["extrinsic"] = {'R':camera['rotation'], 'T':camera['translation']} # {'R', 'T'}
            # sample["intrinsic"] = cam_param['intrinsics_wo_distortion'] # {'c', 'f'}
            sample["p3d_gt"]    = torch.from_numpy(p3d_gt[start_frame:stop_frame])
            sample["trans"]     = torch.from_numpy(trans[start_frame:stop_frame])
            sample["betas"]     = torch.from_numpy(betas[start_frame:stop_frame])
            sample["thetas"]    = torch.from_numpy(thetas[start_frame:stop_frame])
            if action in train_act: trainset.append(sample)
            if action in test_act: testset.append(sample)

        with open(os.path.join(save_path, "trainset.pkl"), 'wb') as f:
            pickle.dump(trainset, f)
        with open(os.path.join(save_path, "testset.pkl"), 'wb') as f:
            pickle.dump(testset, f)
        
    print('Total training/testing samples: {:d}/{:d} \n'.format(len(trainset),len(testset)))
    return trainset, testset