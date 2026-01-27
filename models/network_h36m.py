import torch
import torch.nn as nn
from pytorch3d import transforms
import models.misc as utils
import os

class InitNet(nn.Module):
    def __init__(self, hid_dim: int):
        super(InitNet, self).__init__()
        state_dim = 25
        self.module1 = nn.Sequential(
            nn.Linear(state_dim*3+45*3, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, state_dim*3),
            nn.Tanh(),
        )
        self.module2 = nn.Sequential(
            nn.Linear(45*3+10, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, 10),
            nn.Tanh(),
        )
        self.module3 = nn.Linear(state_dim*3, state_dim*3)
        with torch.no_grad():
            self.module1.apply(self.init_weights)
            self.module2.apply(self.init_weights)
            self.module3.apply(self.init_weights)
        
    def forward(self, t01: torch.Tensor, q01: torch.Tensor, beta0: torch.Tensor, aa0: torch.Tensor, p0: torch.Tensor):
        N, T, V, C   = q01.shape # [N, 2, 25, 3]
        
        inp         = torch.cat((t01[:,0], aa0.reshape(N,-1), p0.reshape(N,-1)), dim=-1)
        aa0_fix     = self.module1(inp).reshape(N,25,-1)
        m0          = t01[:,0] + aa0_fix[:,0]
        aa0_new     = utils.masking(aa0 + aa0_fix[:,1:])
        q0          = transforms.axis_angle_to_quaternion(aa0_new)
        
        beta_fix    = self.module2(torch.cat((p0.reshape(N,-1), beta0), dim=-1))
        
        error_t     = t01[:,1] - t01[:,0]
        error_q     = utils.get_quaternion_error(q01[:,0], q01[:,1])
        error       = torch.cat((error_t, error_q.reshape(N,-1)), dim=-1)
        vel0        = self.module3(error).reshape(N,25,-1)
        v0          = vel0[:,0]
        w0          = utils.masking(vel0[:,1:])
        return [m0, v0, q0, w0], beta_fix
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, a=-1e-2, b=1e-2)
            nn.init.zeros_(m.bias)
            
    def save(self, path, name):
        filtered_dict = {k: v for k, v in self.state_dict().items() if 'tmp_var' not in k} # do not save unnecessary tmp variables..
        os.makedirs(path, exist_ok=True)
        torch.save({'net': filtered_dict}, os.path.join(path, name))
        print("weights saved at: ", os.path.join(path, name))

    def load(self, path, name, device):
        state_dicts = torch.load(os.path.join(path, name), map_location=device)
        network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
        self.load_state_dict(network_state_dict)
        print("weights of trained model loaded")



class CtrlNet(nn.Module):
    def __init__(self, args, hid_dim: int, J_reg: torch.Tensor, dropout_p: float=0.2):
        super(CtrlNet, self).__init__()
        
        "Backbone network"
        state_dim    = 25
        self.network = nn.Sequential(
            nn.Linear(9+(state_dim-1)*11, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.LeakyReLU(),
        )
        
        self.exact = args.exact
        if args.exact: Kp = 40*torch.ones(1,state_dim,1)
        else: Kp = 1000*torch.ones(1,state_dim,1)
        Kp[:,0] = 1
        Kp[:,[8,9,11,12,21,22,23,24]] *= 0. # end joints
        self.Kp = nn.Parameter(Kp, requires_grad=False)
        self.linear_kp  = nn.Linear(hid_dim, state_dim*3)
        
        Kd      = 30*torch.ones(1,state_dim,1)
        Kd[:,0] = 1
        Kd[:,[8,9,11,12,21,22,23,24]] *= 0. # end joints
        self.Kd = nn.Parameter(Kd, requires_grad=False)
        self.linear_kd  = nn.Linear(hid_dim, state_dim*3)
        self.linear_off = nn.Linear(hid_dim, state_dim*3)
        with torch.no_grad():
            nn.init.zeros_(self.linear_off.weight)
            nn.init.zeros_(self.linear_off.bias)
                
        self.second = args.second
        if self.second:
            Ka = 40*torch.ones(1,state_dim,1)
            Ka[:,0], Ka[:,1] = 0., 3.  # trans, g.rot
            Ka[:,[8,9,11,12,21,22,23,24]] *= 0. # end joints
            self.Ka = nn.Parameter(Ka, requires_grad=False)
            self.linear_ka  = nn.Linear(hid_dim, state_dim*3)
        
        self.dt         = 1/25 # real-time motion capture rate
        non_zeros       = torch.nonzero(J_reg, as_tuple=True)
        row_indices, col_indices = non_zeros[0], non_zeros[1]
        self.grouped_indices, self.masks = [], []
        for i in range(row_indices.max().item() + 1):
            mask    = row_indices == i
            self.grouped_indices.append(col_indices[mask])
            self.masks.append(mask)
        self.new_Js     = nn.Parameter(J_reg[non_zeros], requires_grad=True)
        self.J_reg      = J_reg
        self.dropout_p  = dropout_p
        
    def forward(self, x_t: tuple, x_i: tuple, t: int):
        m_in, q_in  = x_i   # N, 3 - N, 24, 4
        m, v, q, w, em, eq = x_t   # N, 3 - N, 24, 4
        N, J, _     = q.shape
        
        # Backbone
        inp         = torch.cat((m, m_in[:,-1], v, q.reshape(N,-1), q_in[:,-1].reshape(N,-1), w.reshape(N,-1)), dim=-1)
        embed       = self.network(inp)
        kp          = self.Kp.repeat(N,1,1) * torch.sigmoid(self.linear_kp(embed).reshape(N,25,3))
        kd          = self.Kd.repeat(N,1,1) * torch.sigmoid(self.linear_kd(embed).reshape(N,25,3))
        offset      = torch.tanh(self.linear_off(embed).reshape(N,25,3))
        
        # Translation
        m_error     = m_in[:,1] - m
        dv          = 200 * (kp[:,0] * m_error - kd[:,0] * v + offset[:,0]) 
        vt          = v + dv * self.dt
        mt          = m + vt * self.dt
        
        # Rotations
        q_error     = utils.get_quaternion_error(q, q_in[:,1])
        q_in_error  = utils.get_quaternion_error(q_in[:,0], q_in[:,1])
        outlier     = torch.any(torch.abs(q_error) > 0.8, dim=-1, keepdim=True)
        q_error[outlier.expand(N,-1,3)] = 0. # reflection
        q_in_error[outlier.expand(N,-1,3)] = 0.
        if self.dropout_p > 0. and self.training:
            dropout_mask = (torch.rand(1,24,3, device=q_error.device) > self.dropout_p).float()
            q_error     = q_error * dropout_mask.repeat(N,1,1)
            
        if not self.second: deriv_2nd = 0
        else:
            ka          = self.Ka.repeat(N,1,1) * torch.sigmoid(self.linear_ka(embed).reshape(N,25,3))
            deriv_2nd   = ka[:,1:]*(q_in_error - eq) # second-order derivative
        dw          = (kp[:,1:]*q_error - kd[:,1:]*w) + (offset[:,1:]) + (deriv_2nd) 
        wt          = w + utils.masking(dw) * self.dt
        wt          = torch.clamp(wt, -25, 25)
        
        if self.exact:
            wt_quat     = utils.axis_angle_to_quaternion(wt, self.dt)
            qt          = transforms.quaternion_raw_multiply(wt_quat, q)
        else:
            wt_quat     = torch.cat((torch.zeros_like(wt[...,0,None]), wt), dim=-1) 
            qt          = q + 0.5 * transforms.quaternion_raw_multiply(wt_quat, q) * self.dt
            qt          = qt / torch.linalg.norm(qt, dim=-1, keepdim=True)
        
        return mt, vt, qt, wt, m_error, q_in_error
    
    def get_new_regressor(self, ):
        J_reg_new   = torch.zeros_like(self.J_reg)
        for i, group in enumerate(self.grouped_indices):
            new_Js  = self.new_Js[self.masks[i]]
            J_reg_new[i, group] = new_Js / new_Js.sum()
        return J_reg_new
    
    def save(self, path, name):
        filtered_dict = {k: v for k, v in self.state_dict().items() if 'tmp_var' not in k} # do not save unnecessary tmp variables..
        os.makedirs(path, exist_ok=True)
        torch.save({'net': filtered_dict}, os.path.join(path, name))
        print("weights saved at: ", os.path.join(path, name))

    def load(self, path, name, device):
        state_dicts = torch.load(os.path.join(path, name), map_location=device)
        network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
        self.load_state_dict(network_state_dict)
        print("weights of trained model loaded")
