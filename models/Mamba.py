# PyTorch相关的库 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset 
from torch.nn import functional as F 
from einops import rearrange 
from tqdm import tqdm 
# 系统相关的库 
import math 
import os 
import urllib.request 
from zipfile import ZipFile 
from transformers import AutoTokenizer 
torch.autograd.set_detect_anomaly(True)

# 配置标识和超参数 
USE_MAMBA = 1 
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0 
# 设定所用设备 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 人为定义的超参数 
d_model = 8
state_size = 128 # 状态大小
seq_len = 100 # 序列长度
batch_size = 256 # 批次大小
current_batch_size = batch_size
different_batch_size = False
h_new = None
temp_buffer = None
    
class S6(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(S6, self).__init__()
        
        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)
        self.fc3 = nn.Linear(d_model, state_size, device=device)
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size
        
        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)

        self.delta = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)
        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)

         # h [batch_size, seq_len, d_model, state_size]
        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)

    def discretization(self):

        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)

        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))

        return self.dA, self.dB

    def forward(self, x):
         # Algorithm 2 MAMBA paper
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))

        self.discretization()

        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:  

            global current_batch_size
            current_batch_size = x.shape[0]

            if self.h.shape[0] != current_batch_size:
                different_batch_size = True

                h_new =  torch.einsum('bldn,bldn->bldn', self.dA, self.h[:current_batch_size, ...]) + rearrange(x, "b l d -> b l d 1") * self.dB

            else:
                different_batch_size = False
                h_new =  torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB

             # y [batch_size, seq_len, d_model]
            self.y = torch.einsum('bln,bldn->bld', self.C, h_new)

            global temp_buffer
            temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()

            return self.y

        else:  
             # h [batch_size, seq_len, d_model, state_size]
            h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
            y = torch.zeros_like(x)
            
            h =  torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB

            
            # y [batch_size, seq_len, d_model]
            y = torch.einsum('bln,bldn->bld', self.C, h)

            return y
class MambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(MambaBlock, self).__init__()

        self.inp_proj = nn.Linear(d_model, 2*d_model, device=device)
        self.out_proj = nn.Linear(2*d_model, d_model, device=device)

        # For residual skip connection
        self.D = nn.Linear(d_model, 2*d_model, device=device)

        # Set _no_weight_decay attribute on bias
        self.out_proj.bias._no_weight_decay = True

        # Initialize bias to a small constant value
        nn.init.constant_(self.out_proj.bias, 1.0)

        self.S6 = S6(seq_len, 2*d_model, state_size, device)

        # Add 1D convolution with kernel size 3
        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1, device=device)

        # Add linear layer for conv output
        self.conv_linear = nn.Linear(2*d_model, 2*d_model, device=device)

        # rmsnorm
        self.norm = RMSNorm(d_model, device=device)

    def forward(self, x):
        """
        x_proj.shape = torch.Size([batch_size, seq_len, 2*d_model])
        x_conv.shape = torch.Size([batch_size, seq_len, 2*d_model])
        x_conv_act.shape = torch.Size([batch_size, seq_len, 2*d_model])
        """
        # Refer to Figure 3 in the MAMBA paper

        x = self.norm(x)
        x_proj = self.inp_proj(x)

        # Add 1D convolution with kernel size 3
        x_conv = self.conv(x_proj)

        x_conv_act = F.silu(x_conv)

        # Add linear layer for conv output
        x_conv_out = self.conv_linear(x_conv_act)

        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)  # Swish activation can be implemented as x * sigmoid(x)

        # residual skip connection with nonlinearity introduced by multiplication
        x_residual = F.silu(self.D(x))

        x_combined = x_act * x_residual

        x_out = self.out_proj(x_combined)

        return x_out
    
class Mamba(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(Mamba, self).__init__()
        self.mamba_block1 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block2 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block3 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block4 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block5 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block6 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block7 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block8 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block9 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block10 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block11 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block12 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block13 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block14 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block15 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block16 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block17 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block18 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block19 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block20 = MambaBlock(seq_len, d_model, state_size, device)
        self.norm = RMSNorm(d_model, device=device)

    def forward(self, x):
        x = self.norm(x)
        x = self.mamba_block1(x)
        x = self.mamba_block2(x)
        x = self.mamba_block3(x)
        x = self.mamba_block4(x) 
        x = self.mamba_block5(x) 
        x = self.mamba_block6(x) 
        x = self.mamba_block7(x) 
        x = self.mamba_block8(x) 
        x = self.mamba_block9(x) 
        x = self.mamba_block10(x) 
        x = self.mamba_block11(x) 
        x = self.mamba_block12(x) 
        x = self.mamba_block13(x) 
        x = self.mamba_block14(x) 
        x = self.mamba_block15(x)
        # x = self.mamba_block16(x)
        # x = self.mamba_block17(x)
        # x = self.mamba_block18(x)
        # x = self.mamba_block19(x)
        # x = self.mamba_block20(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self,d_model: int,eps: float = 1e-5,device: str ='cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

# x= torch.rand(batch_size, seq_len, d_model, device=device)
#  # Create the Mamba model
# mamba = Mamba(seq_len, d_model, state_size, device)

#  # rmsnorm
# norm = RMSNorm(d_model)
# x = norm(x)

#  # Forward pass
# test_output = mamba(x)
# print(f"test_output.shape = {test_output.shape}")  # Should be [batch_size, seq_len, d_model]