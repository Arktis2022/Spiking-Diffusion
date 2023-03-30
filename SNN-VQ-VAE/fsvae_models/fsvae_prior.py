from argparse import ZERO_OR_MORE
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T

import global_v as glv

from .snn_layers import *

# 先验z生成器
class PriorBernoulliSTBP(nn.Module):
    def __init__(self, k=20) -> None:
        
        # 先验z根据之前的输入z来生成目前的z
        # 得到的先验概率，即按理来说应该是这样
        """
        modeling of p(z_t|z_<t)
        """
        
        
        super().__init__()
        self.channels = glv.network_config['latent_dim']
        self.k = k
        self.n_steps = glv.network_config['n_steps']
        
        # 这个东西由全连接层组成
        self.layers = nn.Sequential(
            tdLinear(self.channels,
                    self.channels*2,
                    bias=True,
                    bn=tdBatchNorm(self.channels*2, alpha=2), 
                    spike=LIFSpike()),
            tdLinear(self.channels*2,
                    self.channels*4,
                    bias=True,
                    bn=tdBatchNorm(self.channels*4, alpha=2),
                    spike=LIFSpike()),
            tdLinear(self.channels*4,
                    self.channels*k,
                    bias=True,
                    bn=tdBatchNorm(self.channels*k, alpha=2),
                    spike=LIFSpike())
        )
        
        # .register_buffer：该方法的作用是定义一组参数，该组参数的特别之处在于：模型训练时不会更新（即调用 optimizer.step() 后该组参数不会变化，只可人为地改变它们的值），但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存
        # 因此这个initial_input被固定为形状为(1,c,1)的全0值，作为z0
        self.register_buffer('initial_input', torch.zeros(1, self.channels, 1))# (1,C,1)


    def forward(self, z, scheduled=False, p=None):
        if scheduled:
            return self._forward_scheduled_sampling(z, p)
        else:
            return self._forward(z)
    
    def _forward(self, z):
        
        # 前向传播，从先验分布中进行采样，说是这么说，采样其实也是通过神经网络进行的
        # 输入BCT，变成BCkT
        """
        input z: (B,C,T) # latent spike sampled from posterior
        output : (B,C,k,T) # indicates p(z_t|z_<t) (t=1,...,T)
        """
        z_shape = z.shape # (B,C,T)
        batch_size = z_shape[0]
        z = z.detach()
        
        # 初始化的z_0，本来形状是(1,c,1),扩展为(B,C,1)
        z0 = self.initial_input.repeat(batch_size, 1, 1) # (B,C,1)
        
        # 将输入z0与z进行拼接
        # 即将形状(B,C,1)的z0与形状为(B,C,T-1)的latent_x组合
        # 这里z[...,:-1]选取的是最后一个维度的除了最后一个元素的所有元素
        inputs = torch.cat([z0, z[...,:-1]], dim=-1) # (B,C,T)
        
        # 输出根据input得到
        outputs = self.layers(inputs) # (B,C*k,T)
        
        # 先验z即将上述outputs堆叠得到，为什么选择k？
        p_z = outputs.view(batch_size, self.channels, self.k, self.n_steps) # (B,C,k,T)
        return p_z

    def _forward_scheduled_sampling(self, z, p):
        """
        use scheduled sampling
        input 
            z: (B,C,T) # latent spike sampled from posterior
            p: float # prob of scheduled sampling
        output : (B,C,k,T) # indicates p(z_t|z_<t) (t=1,...,T)
        """
        z_shape = z.shape # (B,C,T)
        batch_size = z_shape[0]
        z = z.detach()

        z_t_minus = self.initial_input.repeat(batch_size,1,1) # z_<t, z0=zeros:(B,C,1)
        if self.training:
            with torch.no_grad():
                for t in range(self.n_steps-1):
                    if t>=5 and random.random() < p: # scheduled sampling                    
                        outputs = self.layers(z_t_minus.detach()) #binary (B, C*k, t+1) z_<=t
                        p_z_t = outputs[...,-1] # (B, C*k, 1)
                        # sampling from p(z_t | z_<t)
                        prob1 = p_z_t.view(batch_size, self.channels, self.k).mean(-1) # (B,C)
                        prob1 = prob1 + 1e-3 * torch.randn_like(prob1) 
                        z_t = (prob1>0.5).float() # (B,C)
                        z_t = z_t.view(batch_size, self.channels, 1) #(B,C,1)
                        z_t_minus = torch.cat([z_t_minus, z_t], dim=-1) # (B,C,t+2)
                    else:
                        z_t_minus = torch.cat([z_t_minus, z[...,t].unsqueeze(-1)], dim=-1) # (B,C,t+2)
        else: # for test time
            z_t_minus = torch.cat([z_t_minus, z[:,:,:-1]], dim=-1) # (B,C,T)

        z_t_minus = z_t_minus.detach() # (B,C,T) z_{<=T-1} 
        p_z = self.layers(z_t_minus) # (B,C*k,T)
        p_z = p_z.view(batch_size, self.channels, self.k, self.n_steps)# (B,C,k,T)
        return p_z

    def sample(self, batch_size=64):
        z_minus_t = self.initial_input.repeat(batch_size, 1, 1) # (B, C, 1)
        for t in range(self.n_steps):
            outputs = self.layers(z_minus_t) # (B, C*k, t+1)
            p_z_t = outputs[...,-1] # (B, C*k, 1)

            random_index = torch.randint(0, self.k, (batch_size*self.channels,)) \
                            + torch.arange(start=0, end=batch_size*self.channels*self.k, step=self.k) #(B*C,) pick one from k
            random_index = random_index.to(z_minus_t.device)

            z_t = p_z_t.view(batch_size*self.channels*self.k)[random_index] # (B*C,)
            z_t = z_t.view(batch_size, self.channels, 1) #(B,C,1)
            z_minus_t = torch.cat([z_minus_t, z_t], dim=-1) # (B,C,t+2)

        
        sampled_z = z_minus_t[...,1:] # (B,C,T)

        return sampled_z

