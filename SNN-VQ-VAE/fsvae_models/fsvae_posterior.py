import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import global_v as glv

from .snn_layers import *

# 后验z生成，应该是要先看这个
class PosteriorBernoulliSTBP(nn.Module):
    def __init__(self, k=20) -> None:
        
        # 根据x<=t以及之前的z得到这一步应该有的z，即为后验z
        # 看上去很高大上，但是感觉就是把x编码得到的 与 z0和z[0:t-1]拼接得到的
        # 这俩给到全连接网络
        # 得到采样输出
        """
        modeling of q(z_t | x_<=t, z_<t)
        """
        super().__init__()
        self.channels = glv.network_config['latent_dim']
        self.k = k
        self.n_steps = glv.network_config['n_steps']
        
        # 网络形状：将输入变为k倍
        # 输入形状 B C T（batch_size, channels,num_steps）
        # 这个版本的snn就是把时间当初一个新维度
        self.layers = nn.Sequential(
            tdLinear(self.channels*2,
                    self.channels*2,
                    bias=True,
                    bn=tdBatchNorm(self.channels*2, alpha=2), 
                    spike=LIFSpike()),
            # 这里的tdLinear已经集成了tdBN方法以及LIF激活函数
            
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
        # z0的值并不重要 也可以是随机初始化的 但为了结果可以复现 选择了全0
        self.register_buffer('initial_input', torch.zeros(1, self.channels, 1))# (1,C,1)

        self.is_true_scheduled_sampling = True

    def forward(self, x):
        """
        input: 
            x:(B,C,T)
        returns: 
            sampled_z:(B,C,T)
            q_z: (B,C,k,T) # indicates q(z_t | x_<=t, z_<t) (t=1,...,T)
        """
        
        # 输入x，这个x是编码器编码再线性化再全连接得到的latent_x
        x_shape = x.shape # (B,C,T)
        
        # 得到batch_size
        batch_size=x_shape[0]
        random_indices = []
        # 从分布中采样z？
        with torch.no_grad():
            
            # 这里得到z0，即值全0，形状为(B,T,1)
            z_t_minus = self.initial_input.repeat(x_shape[0],1,1) # z_<t z0=zeros:(B,C,1)
            
            # 对于前T-1个时间步
            for t in range(self.n_steps-1):
                # 输入即为latent_x（B,C,T）与z0（B,C,1）的拼接
                # 得到的形状为（B,2C,t+1）
                inputs = torch.cat([x[...,0:t+1].detach(), z_t_minus.detach()], dim=1) # (B,C+C,t+1) x_<=t and z_<t
 
                # 将这一堆给到layers，得到形状为(B,kC,t+1)的东西
                outputs = self.layers(inputs) #(B, C*k, t+1) 
                
                # 而后验z的时间t下的值取到了(B,k*C,1)，即上面的input的最后一个
                q_z_t = outputs[...,-1] # (B, C*k, 1) q(z_t | x_<=t, z_<t) 
                
                # 从上面得到的(B,k*C,1)随机采样 q(z_t | x_<=t, z_<t)
                # 从0开始的k个中依次采样一个
                random_index = torch.randint(0, self.k, (batch_size*self.channels,)) \
                            + torch.arange(start=0, end=batch_size*self.channels*self.k, step=self.k) #(B*C,) select 1 from every k value
                random_index = random_index.to(x.device)
                random_indices.append(random_index)
                
                # 采样结束后得到形状为(B*C)的东西
                z_t = q_z_t.view(batch_size*self.channels*self.k)[random_index] # (B*C,)
                
                # 最后resize为(b,c,1)
                z_t = z_t.view(batch_size, self.channels, 1) #(B,C,1)
                
                # 将z_t-1更新一下
                z_t_minus = torch.cat([z_t_minus, z_t], dim=-1) # (B,C,t+2)
                # 因此z_t-1确实是逐渐延长的

        # 最后得到的z形状则为(B,C,T)
        # 再将这个最后的z输入到layers中得到最后需要的(B,C*k,T)
        z_t_minus = z_t_minus.detach() # (B,C,T) z_0,...,z_{T-1}
        q_z = self.layers(torch.cat([x, z_t_minus], dim=1)) # (B,C*k,T)
        
        # input z_t_minus again to calculate tdBN
        sampled_z = None
        for t in range(self.n_steps):
            
            if t == self.n_steps-1:
                # when t=T
                random_index = torch.randint(0, self.k, (batch_size*self.channels,)) \
                            + torch.arange(start=0, end=batch_size*self.channels*self.k, step=self.k)
                random_indices.append(random_index)
            else:
                # when t<=T-1
                random_index = random_indices[t]

            # sampling
            sampled_z_t = q_z[...,t].view(batch_size*self.channels*self.k)[random_index] # (B*C,)
            sampled_z_t = sampled_z_t.view(batch_size, self.channels, 1) #(B,C,1)
            if t==0:
                sampled_z = sampled_z_t
            else:
                # sampled_z由T个B,C,1组成
                # 方法是，本来的q_z是b,c*k,t
                # 每个时间步中，从q_z[:,t](B,C*K,1)中采样得到（B,C,1）
                # 将每个时间步的采样结果连接，得到形状（B,C,T）的sampled_z
                # 最后一个维度从1增加到T
                sampled_z = torch.cat([sampled_z, sampled_z_t], dim=-1)
        
        # q_z
        q_z = q_z.view(batch_size, self.channels, self.k, self.n_steps)# (B,C,k,T)
        
        # 最后sampled_z的形状：[250, 128, 16]——B,C,T
        # 最后q_z的形状：[250, 128, 20, 16]——B,C,K,T
        return sampled_z, q_z