import os
import sys
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader
import torch.distributions as dists

import torchvision
from torchvision import datasets, transforms


from spikingjelly.activation_based import neuron, functional ,layer, surrogate, monitor
from spikingjelly import visualizing
import matplotlib.pyplot as plt

from .vae_model import *

def get_data_for_diff(train_loader,model):
    print('prepare data for train diffusion...')
    model.eval()
    train_indices = []
    for images, labels in train_loader:
        images = images - 0.5 # normalize to [-0.5, 0.5]
        images = images.cuda()
        images_spike = images.unsqueeze(0).repeat(16, 1, 1, 1, 1)

        with torch.inference_mode():
            _,_,encoding_indices = model(images_spike,images) # [B, C, H, W, T]
            train_indices.append(encoding_indices.reshape(images.shape[0],7,7).cpu())
        #break
    return train_indices  

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        pass

class AbsorbingDiffusion(Sampler):
    def __init__(self, denoise_fn, mask_id):
        super().__init__()
        self.num_classes = denoise_fn.num_embeddings
        self.shape = [7,7]
        self.num_timesteps = 49
        self.mask_id = mask_id
        self._denoise_fn = denoise_fn
        self.n_samples = 16
        self.mask_schedule = 'random'
        self.loss_type = 'reweighted_elbo'


    def sample_time(self, b, device):
        t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()
        pt = torch.ones_like(t).float() / self.num_timesteps
        return t, pt

    def q_sample(self, x_0, t):
        x_t, x_0_ignore = x_0.clone(), x_0.clone()
        t_mask = t.reshape(x_0.shape[0], 1, 1,1)
        t_mask = t_mask.expand(x_0.shape[0], 1, 7, 7) # 和x形状相同进行掩码    
        mask = torch.rand_like(x_t.float()) < (t_mask.float() / self.num_timesteps)

        # 将x_t中mask为1的位置替换为self.mask_id，这是一个特殊的标识符，表示该位置被掩码了。
        x_t[mask] = self.mask_id

        # 将x_0_ignore中mask为0的位置替换为-1，这是一个忽略索引，表示该位置不参与损失计算
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask

    
    def _train_loss(self, x_0):

        b, device = x_0.size(0), x_0.device
        
        t, pt = self.sample_time(b, device)

        x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t)

        x_0_hat_logits = self._denoise_fn(x_t, t=t)

        cross_entropy_loss = F.cross_entropy(x_0_hat_logits.reshape(b,self.num_classes,49), x_0_ignore.reshape(b,49).type(torch.LongTensor).to(device), 
        ignore_index=-1, reduction='none').sum(1)
        
        vb_loss = cross_entropy_loss / t
        vb_loss = vb_loss / pt
        vb_loss = vb_loss / (math.log(2) * x_0.shape[1:].numel())

        if self.loss_type == 'elbo':
            loss = vb_loss
        elif self.loss_type == 'reweighted_elbo': 
            weight = (1 - (t / self.num_timesteps))
            loss = weight * cross_entropy_loss
            loss = loss / (math.log(2) * x_0.shape[1:].numel())
        else:
            raise ValueError

        return loss.mean()

    def sample(self, temp=1.0, sample_steps=None):

        b, device = int(self.n_samples), 'cuda'
        x_t = torch.ones(b,1,7,7, device=device).long() * self.mask_id  
        unmasked = torch.zeros_like(x_t, device=device).bool()
        # 确定sample的次数 一般是7*7  每一次去掉一个掩码
        # 如果小于该数值 则等效于跳步
        sample_steps = list(range(1, sample_steps+1))            
        for t in reversed(sample_steps):
            #print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)

            # where to unmask
            t_mask = t.reshape(b, 1, 1,1)
            t_mask = t_mask.expand(b, 1, 7, 7)
            changes = torch.rand_like(x_t.float()) < 1/t_mask.float()
            changes = changes

            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            # x_t_for_unet = F.pad(input=x_t, pad=(0, 1, 1, 0), mode='constant', value = mask_id) # for unet
            #denoise_fn = self._denoise_fn.cuda(1)
            x_0_logits = self._denoise_fn(x_t.float(), t=t).permute(0,2,3,1)
            functional.reset_net(self._denoise_fn)
            # x_0_logits = x_0_logits[:,:,1:,0:-1] # for unet

            # scale by temperature
            # 温度越高 多样性越强
            x_0_logits = x_0_logits / temp
            # 创建一个类别分布，表示每个token的概率
            x_0_dist = dists.Categorical(
                logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_0_hat = x_0_hat.unsqueeze(dim=1)
            x_t[changes] = x_0_hat[changes]

        return x_t

    def train_iter(self, x):
        loss = self._train_loss(x)
        stats = {'loss': loss}
        return stats
    

class DummyModel(nn.Module):
    """
    This should be transformer, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """
    # batch_size 1 7 7 
    # batch_size  1287 7
    # batch_size 1 7 7 
    def __init__(self, n_channel: int,num_embeddings) -> None:
        super(DummyModel, self).__init__()
        self.num_embeddings = num_embeddings
        self.conv1 = nn.Sequential(  # with batchnorm
            layer.Conv2d(in_channels=n_channel*2, out_channels=64, kernel_size=3, stride=1, padding=1),
            layer.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()))
        self.conv2 = nn.Sequential(
            layer.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            layer.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )
        self.conv3=nn.Sequential(
            layer.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            layer.BatchNorm2d(256),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )
        self.conv4=nn.Sequential(
            layer.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            layer.BatchNorm2d(512),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )
        self.conv5=nn.Sequential(
            layer.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            layer.BatchNorm2d(256),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )
        self.conv6=nn.Sequential(
            layer.Conv2d(256+64, num_embeddings, 3, 1, 1),
        )

    def forward(self, x, t) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.

        # FIXME:
        # x:b,c,h,w
        # t:b
        t = torch.ones_like(x)*(t.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        x = torch.cat((x,t),dim=1)
        #print(x.shape)
        x = x.unsqueeze(dim = 0).repeat(16, 1, 1, 1, 1)
    
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(torch.cat((x5,x1),dim=2))
        outputs = torch.sum(x6,dim = 0)/16

        return outputs


