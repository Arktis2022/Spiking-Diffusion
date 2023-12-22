import os
import sys

import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
import torch.utils.data.dataloader
from torchvision import datasets, transforms

from spikingjelly.activation_based import neuron, functional ,layer, surrogate, monitor
from spikingjelly import visualizing
import matplotlib.pyplot as plt

from .snn_layers import *

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim 
        self.num_embeddings = num_embeddings 
        self.commitment_cost = commitment_cost 
        self.memout = MembraneOutputLayer()
        self.num_step = 16
        self.psp = PSP()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        #self.alpha=1
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.poisson = nn.Sequential(
            layer.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            layer.BatchNorm2d(16),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        # (T,N,C,H,W)->(N,C,H,W) 16 128 16 7 7
        x_memout = (1-self.alpha)*self.memout(x) + self.alpha*torch.sum(x,dim=0)/self.num_step
        # [128, 16, 7, 7] -> [128, 7, 7,16]
        x_memout = x_memout.permute(0, 2, 3, 1).contiguous()
        # [128, 7, 7,16] -> [6272, 16]
        flat_x = x_memout.reshape(-1, self.embedding_dim)
    
        encoding_indices = self.get_code_indices(flat_x)

        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x_memout) # [128, 7, 7, 16]
        
        if not self.training:
            quantized = quantized.permute(0, 3, 1, 2).contiguous() #[128, 16, 7, 7]
            quantized = torch.unsqueeze(quantized, dim=0) #[1, 128, 16, 7, 7]
            quantized = quantized.repeat(16, 1, 1, 1, 1) #[16, 128, 16, 7, 7]
            quantized = self.poisson(quantized) # [16, 128, 16, 7, 7]
            return quantized,encoding_indices


        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x_memout.detach())

        # commitment loss
        e_latent_loss = F.mse_loss(x_memout, quantized.detach())

        loss_1 = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator 
        # forward quantized = quantized
        # backward quantized = x
        quantized = x_memout + (quantized - x_memout).detach()
        # [128, 7, 7,16]->[128,16,7,7]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # FIXME:
        quantized = torch.unsqueeze(quantized, dim=0) # [1,128,16,7,7] 
        quantized = quantized.repeat(16, 1, 1, 1, 1) # [16,128,16,7,7]
        quantized = self.poisson(quantized) # [16,128,16,7,7]

        q_latent_loss_2 = torch.mean((self.psp(quantized)-self.psp(x.detach()))**2)
        e_latent_loss_2 = torch.mean((self.psp(quantized.detach())-self.psp(x))**2)
        loss_2 =  q_latent_loss_2 + self.commitment_cost * e_latent_loss_2
        
        return quantized, loss_1 +loss_2
    
    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.weight.t())
        ) 
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)      

class Encoder(nn.Module):
    """Encoder of VQ-VAE"""
    
    def __init__(self, in_dim=1, latent_dim=16):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        self.snn_convs = nn.Sequential(
            layer.Conv2d(in_channels=in_dim, out_channels=32, kernel_size=3,
            stride=2, padding=1),
            layer.BatchNorm2d(32),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            layer.Conv2d(in_channels=32,out_channels=64, kernel_size=3,
            stride=2, padding=1),
            layer.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            layer.Conv2d(in_channels=64, out_channels=latent_dim, kernel_size=1,
            stride=1, padding=0),
            layer.BatchNorm2d(latent_dim),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            )
        
    def forward(self, x):
        # [t, b, c, h, w]
        x = self.snn_convs(x)
        return x

class Decoder(nn.Module):
    """Decoder of VQ-VAE"""
    
    def __init__(self, out_dim=1, latent_dim=16):
        super().__init__()
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        
        self.snn_convs = nn.Sequential(
            layer.ConvTranspose2d(in_channels=latent_dim, out_channels=64, kernel_size=3,
            stride=2, padding=1, output_padding=1),
            layer.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),


            layer.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3,
            stride=2, padding=1, output_padding=1),
            layer.BatchNorm2d(32),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),


            layer.ConvTranspose2d(in_channels=32, out_channels=out_dim, kernel_size=3,
            stride=1, padding=1, output_padding=0),

        )
    def forward(self, x):
        # [t, b, c, h, w]
        x = self.snn_convs(x)
        return x

class SNN_VQVAE(nn.Module):

    """VQ-VAE"""
    
    def __init__(self, in_dim, embedding_dim, num_embeddings, data_variance, 
        commitment_cost=0.25):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.data_variance = data_variance
        
        self.encoder = Encoder(in_dim, embedding_dim)
        self.vq_layer = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)
        self.decoder = Decoder(in_dim, embedding_dim)
        # TODO:
        self.memout = MembraneOutputLayer()

    def forward(self, x,image):
        # x: [t, B, C, H, W]
        z = self.encoder(x)
        # print(z.shape)
        if not self.training:
            e,enco = self.vq_layer(z)
            x_recon = self.decoder(e)
            x_recon = torch.tanh(self.memout(x_recon))
            return e, x_recon,enco
        
        e, e_q_loss = self.vq_layer(z)
        x_recon = self.decoder(e)
        x_recon = torch.tanh(self.memout(x_recon))
        
        real_recon_loss = F.mse_loss(x_recon, image)
        recon_loss = real_recon_loss / self.data_variance
        
        return e_q_loss,recon_loss,real_recon_loss

class SNN_VAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        in_channels = 1
        latent_dim = 28*2
        self.latent_dim = latent_dim
        self.n_steps = 16

        self.k = 20
        modules = []
        is_first_conv = True
        
        self.encoder = Encoder()
        
        self.before_latent_layer = nn.Sequential(
            layer.Linear(in_features=784, out_features=latent_dim),
            ##layer.BatchNorm1d(latent_dim),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )

        self.prior = PriorBernoulliSTBP(self.k)
        self.posterior = PosteriorBernoulliSTBP(self.k)
        self.decoder_input = nn.Sequential(
            layer.Linear(in_features=latent_dim, out_features=16*7*7),
            #layer.BatchNorm1d(16*7*7),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )

        self.decoder = Decoder()

        self.p = 0
        
        self.membrane_output_layer = MembraneOutputLayer()
        self.psp = PSP()


    def encode(self, x, scheduled=True):
        # print(x.shape)
        x = self.encoder(x) # (T,N,C,7,7)

        x = torch.flatten(x, start_dim=2, end_dim=-1) # (T,N,C*H*W)
        # print(x.shape)

        latent_x = self.before_latent_layer(x) # (T,N,latent_dim)
        # print(latent_x.shape)

        sampled_z, q_z = self.posterior(latent_x) # sampled_z:(T,B,C,1,1), q_z:(T,B,C,k)
        #print(sampled_z.shape,q_z.shape)

        p_z = self.prior(sampled_z, scheduled, self.p)
        #print(p_z.shape)

        return sampled_z, q_z, p_z


    def decode(self, z):
        result = self.decoder_input(z) # (T,N,C*H*W)
        #print(result.shape)

        result = result.view(self.n_steps, result.shape[1], 16, 7, 7) # (T,N,C,H,W) = 16 128 16 7 7 
        #print(result.shape)

        result = self.decoder(result)# (T,N,C,H,W)
        #print(result.shape)

        #result = self.final_layer(result)# (T,N,C,H,W)
        out = torch.tanh(self.membrane_output_layer(result))        
        return out

    def sample(self, batch_size=64):
        sampled_z = self.prior.sample(batch_size)
        sampled_x = self.decode(sampled_z)
        return sampled_x, sampled_z
    
    def loss_function_mmd(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (T,N,latent_dim,k)
        p_z is p(z): (T,N,latent_dim,k)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        q_z_ber = torch.mean(q_z, dim=-1) # (T, N, latent_dim)
        p_z_ber = torch.mean(p_z, dim=-1) # (T, N, latent_dim)

        #kld_loss = torch.mean((q_z_ber - p_z_ber)**2)
        mmd_loss = torch.mean((self.psp(q_z_ber)-self.psp(p_z_ber))**2)
        loss = recons_loss + mmd_loss
        return mmd_loss,recons_loss
    
    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4,4)

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p-init_p) * epoch / max_epoch + init_p


    def forward(self, x, image, scheduled=True):
        sampled_z, q_z, p_z = self.encode(x, scheduled)
        x_recon = self.decode(sampled_z)
        #print(x_recon.shape)
        if not self.training:
            return sampled_z, x_recon
        
        loss_mmd,loss_rec = self.loss_function_mmd(image, x_recon, q_z, p_z)
        return loss_mmd,loss_rec
class PriorBernoulliSTBP(nn.Module):
    def __init__(self, k=20) -> None:

        """
        modeling of p(z_t|z_<t)
        """

        super().__init__()
        self.channels = 28*2
        self.k = k
        self.n_steps = 16
        

        self.layers = nn.Sequential(
            layer.Linear(self.channels, self.channels*2),
            #layer.BatchNorm1d(self.channels*2),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(self.channels*2, self.channels*4),
            #layer.BatchNorm1d(self.channels*4),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),          

            layer.Linear(self.channels*4, self.channels*k),
            #layer.BatchNorm1d(self.channels*k),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),           
        )

        self.register_buffer('initial_input', torch.zeros(1, 1, self.channels))# (1,1,c)


    def forward(self, z, scheduled=True, p=None):
        if scheduled:
            return self._forward_scheduled_sampling(z, p)
        else:
            return self._forward(z)
    
    def _forward(self, z):

        """
        input z: (T,B,C) # latent spike sampled from posterior
        output : (T,B,C,k) # indicates p(z_t|z_<t) (t=1,...,T)
        """
        z_shape = z.shape # (T,B,C)
        batch_size = z_shape[1]
        z = z.detach()
        

        z0 = self.initial_input.repeat(1, batch_size, 1) # (1,B,C)
        

        inputs = torch.cat([z0, z[:-1,...]], dim=0) # (T,B,C)

        # FIXME:
        outputs = self.layers(inputs) # (T, B, C*k)

        p_z = outputs.view(self.n_steps, batch_size, self.channels, self.k) # (T,B,C,k)
        return p_z

    def _forward_scheduled_sampling(self, z, p):
        """
        use scheduled sampling
        input 
            z: (T,B,C) # latent spike sampled from posterior
            p: float # prob of scheduled sampling
        output : (T,B,C,k) # indicates p(z_t|z_<t) (t=1,...,T)
        """
        z_shape = z.shape # (T,B,C)
        batch_size = z_shape[1]
        z = z.detach()

        z_t_minus = self.initial_input.repeat(1,batch_size,1) # z_<t, z0=zeros:(1,B,C)
        if self.training:
            with torch.no_grad():
                for t in range(self.n_steps-1):
                    if t>=5 and random.random() < p: # scheduled sampling
                        # FIXME:      
                        outputs = self.layers(z_t_minus.detach()) # binary (t+1, B, C*k) z_<=t
                        p_z_t = outputs[-1,...] # (1, B, C*k)
                        # sampling from p(z_t | z_<t)
                        prob1 = p_z_t.view(batch_size, self.channels, self.k).mean(-1) # (B,C)
                        prob1 = prob1 + 1e-3 * torch.randn_like(prob1) 
                        z_t = (prob1>0.5).float() # (B,C)
                        z_t = z_t.view(1, batch_size, self.channels) #(1, B, C)
                        z_t_minus = torch.cat([z_t_minus, z_t], dim=0) # (t+2, B, C)
                    else:
                        z_t_minus = torch.cat([z_t_minus, z[t,...].unsqueeze(0)], dim=0) # (t+2, B,C)
        else: # for test time
            #print(z_t_minus.shape)
            #print(z[:-1,:,:].shape)
            z_t_minus = torch.cat([z_t_minus, z[:-1,:,:]], dim=0) # (T,B,C)

        z_t_minus = z_t_minus.detach() # (T,B,C) z_{<=T-1} 
        #print(z_t_minus.shape)
        # FIXME:
        p_z = self.layers(z_t_minus) # (T, B, C*k)
        p_z = p_z.view(self.n_steps, batch_size, self.channels, self.k)# (T,B,C,k)
        return p_z

    def sample(self, batch_size=64):
        z_minus_t = self.initial_input.repeat(1, batch_size, 1) # (1, B, C)
        for t in range(self.n_steps):
            # FIXME:
            outputs = self.layers(z_minus_t) # (B, C*k, t+1)

            p_z_t = outputs[-1,...] # (B, C*k, 1)

            random_index = torch.randint(0, self.k, (batch_size*self.channels,)) \
                            + torch.arange(start=0, end=batch_size*self.channels*self.k, step=self.k) #(B*C,) pick one from k
            random_index = random_index.to(z_minus_t.device)

            z_t = p_z_t.view(batch_size*self.channels*self.k)[random_index] # (B*C,)
            z_t = z_t.view(1, batch_size, self.channels) #(1,B,C)
            z_minus_t = torch.cat([z_minus_t, z_t], dim=0) # (t+2,B,C)

        sampled_z = z_minus_t[1:,...] # (T,B,C)

        return sampled_z

class PosteriorBernoulliSTBP(nn.Module):
    def __init__(self, k=20) -> None:
        

        """
        modeling of q(z_t | x_<=t, z_<t)
        """
        super().__init__()
        self.channels = 28*2
        self.k = k
        self.n_steps = 16
        
        self.layers = nn.Sequential(
            # tdLinear(self.channels*2,
            #         self.channels*2,
            #         bias=True,
            #         bn=tdBatchNorm(self.channels*2, alpha=2), 
            #         spike=LIFSpike()),
            layer.Linear(self.channels*2, self.channels*2),
            #layer.BatchNorm1d(self.channels*2),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            # tdLinear(self.channels*2,
            #         self.channels*4,
            #         bias=True,
            #         bn=tdBatchNorm(self.channels*4, alpha=2),
            #         spike=LIFSpike()),
            layer.Linear(self.channels*2, self.channels*4),
            #layer.BatchNorm1d(self.channels*4),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            # tdLinear(self.channels*4,
            #         self.channels*k,
            #         bias=True,
            #         bn=tdBatchNorm(self.channels*k, alpha=2),
            #         spike=LIFSpike())
            layer.Linear(self.channels*4, self.channels*k),
            #layer.BatchNorm1d(self.channels*k),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )

        self.register_buffer('initial_input', torch.zeros(1, 1, self.channels))# (1,C,1)

        self.is_true_scheduled_sampling = True

    def forward(self, x):
        """
        input: 
            x:(T,B,C)
        returns: 
            sampled_z:(T,B,C)
            q_z: (T,B,C,k) # indicates q(z_t | x_<=t, z_<t) (t=1,...,T)
        """
        

        x_shape = x.shape # (T,B,C) 128 32 16
        

        batch_size=x_shape[1]
        random_indices = []
        
        with torch.no_grad():
            

            z_t_minus = self.initial_input.repeat(1,x_shape[1],1) # z_<t z0=zeros:(1,B,C)

            for t in range(self.n_steps-1):

                # print(x.shape, z_t_minus.shape)
                inputs = torch.cat([x[:t+1,...].detach(), z_t_minus.detach()], dim=-1) # (t+1,B,C+C) x_<=t and z_<t

                outputs = self.layers(inputs) #(t+1, B, C*k) 

                q_z_t = outputs[-1,...] # (1,B,C*k) q(z_t | x_<=t, z_<t) 

                random_index = torch.randint(0, self.k, (batch_size*self.channels,)) \
                            + torch.arange(start=0, end=batch_size*self.channels*self.k, step=self.k) #(B*C,) select 1 from every k value
                random_index = random_index.to(x.device)
                random_indices.append(random_index)

                z_t = q_z_t.view(batch_size*self.channels*self.k)[random_index] # (B*C,)
                
                # 最后resize为(1,b,c)
                z_t = z_t.view(1, batch_size, self.channels) #(1,B,C)
                
                # 将z_t-1更新一下
                # print(z_t.shape, z_t_minus.shape)
                z_t_minus = torch.cat([z_t_minus, z_t], dim=0) # (t+2,B,C)


        z_t_minus = z_t_minus.detach() # (T,B,C) z_0,...,z_{T-1}

        # FIXME:
        inputs = torch.cat([x, z_t_minus], dim=-1) # [T,B,2*C]
        q_z = self.layers(inputs) # (T,B,C*k)
        
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
            sampled_z_t = q_z[t,...].view(batch_size*self.channels*self.k)[random_index] # (B*C,)
            sampled_z_t = sampled_z_t.view(1, batch_size, self.channels) #(1,B,C)
            if t==0:
                sampled_z = sampled_z_t
            else:
                sampled_z = torch.cat([sampled_z, sampled_z_t], dim=0)
        
        # q_z
        q_z = q_z.view(self.n_steps, batch_size, self.channels, self.k)# (T,B,C,k)
        

        return sampled_z, q_z


class CNN_VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        # [B, H, W, C] -> [BHW, C]
        flat_x = x.reshape(-1, self.embedding_dim)
        
        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x) # [B, H, W, C]
        
        if not self.training:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            return quantized,encoding_indices
        
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss
    
    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.weight.t())
        ) # [N, M]
        encoding_indices = torch.argmin(distances, dim=1) # [N,]
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)  

class CNN_Encoder(nn.Module):
    """Encoder of VQ-VAE"""
    
    def __init__(self, in_dim=3, latent_dim=16):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_dim, 1),
        )
        
    def forward(self, x):
        return self.convs(x)

class CNN_Decoder(nn.Module):
    """Decoder of VQ-VAE"""
    
    def __init__(self, out_dim=1, latent_dim=16):
        super().__init__()
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_dim, 3, padding=1),
        )
        
    def forward(self, x):
        return self.convs(x)

class VQVAE(nn.Module):
    """VQ-VAE"""
    
    def __init__(self, in_dim, embedding_dim, num_embeddings, data_variance, 
                 commitment_cost=0.25):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.data_variance = data_variance
        
        self.encoder = CNN_Encoder(in_dim, embedding_dim)
        self.vq_layer = CNN_VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)
        self.decoder = CNN_Decoder(in_dim, embedding_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        if not self.training:
            e,enco = self.vq_layer(z)
            x_recon = self.decoder(e)
            return e, x_recon,enco
        
        e, e_q_loss = self.vq_layer(z)
        x_recon = self.decoder(e)
        
        recon_loss = F.mse_loss(x_recon, x) / self.data_variance
        
        return e_q_loss , recon_loss ,F.mse_loss(x_recon, x)   

class VectorQuantizer_uni(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim 
        self.num_embeddings = num_embeddings 
        self.commitment_cost = commitment_cost 
        self.memout = MembraneOutputLayer()
        self.num_step = 16
        self.psp = PSP()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.poisson = nn.Sequential(
            layer.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            layer.BatchNorm2d(16),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )


    def forward(self, x):
        # (T,N,C,H,W)->(N,C,H,W) 16 128 16 7 7
        x_memout = (1-self.alpha)*self.memout(x) + self.alpha*torch.sum(x,dim=0)/self.num_step
        # [128, 16, 7, 7] -> [128, 7, 7,16]
        x_memout = x_memout.permute(0, 2, 3, 1).contiguous()
        # [128, 7, 7,16] -> [6272, 16]
        flat_x = x_memout.reshape(-1, self.embedding_dim)

        encoding_indices = self.get_code_indices(flat_x)


        #num_indices, _ = torch.unique(encoding_indices,return_counts=True)

        num_indices, _ = torch.unique(encoding_indices,return_counts=True)
        class_indices = torch.bincount(encoding_indices,minlength=self.num_embeddings)
        #print(torch.count_nonzero(class_indices))
        max_index = torch.argmax(class_indices)
        mask = torch.ne(torch.arange(self.num_embeddings).cuda(), max_index)
        taergets_indices = torch.ones(self.num_embeddings,)*len(encoding_indices)/self.num_embeddings
        valid_class_indices = torch.masked_select(class_indices, mask)
        valid_taergets_indices = torch.masked_select(taergets_indices.cuda(), mask.cuda())
        print(len(encoding_indices))
        print(valid_class_indices)
        print(valid_taergets_indices)
        FID_loss = 0.001*F.mse_loss(valid_class_indices.cuda(), valid_taergets_indices.cuda())

        print(num_indices.shape, FID_loss.item())
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x_memout) # [128, 7, 7, 16]
        
        if not self.training:
            quantized = quantized.permute(0, 3, 1, 2).contiguous() #[128, 16, 7, 7]
            quantized = torch.unsqueeze(quantized, dim=0) #[1, 128, 16, 7, 7]
            quantized = quantized.repeat(16, 1, 1, 1, 1) #[16, 128, 16, 7, 7]
            quantized = self.poisson(quantized) # [16, 128, 16, 7, 7]
            return quantized,encoding_indices

        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x_memout.detach())

        # commitment loss
        e_latent_loss = F.mse_loss(x_memout, quantized.detach())

        loss_1 = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator 
        # forward quantized = quantized
        # backward quantized = x
        quantized = x_memout + (quantized - x_memout).detach()
        # [128, 7, 7,16]->[128,16,7,7]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # FIXME:
        quantized = torch.unsqueeze(quantized, dim=0) # [1,128,16,7,7] 
        quantized = quantized.repeat(16, 1, 1, 1, 1) # [16,128,16,7,7]
        quantized = self.poisson(quantized) # [16,128,16,7,7]

        q_latent_loss_2 = torch.mean((self.psp(quantized)-self.psp(x.detach()))**2)
        e_latent_loss_2 = torch.mean((self.psp(quantized.detach())-self.psp(x))**2)
        loss_2 =  q_latent_loss_2 + self.commitment_cost * e_latent_loss_2
        FID_loss=torch.tensor(0)
        return quantized, loss_1 +loss_2,FID_loss

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.weight.t())
        ) 
        encoding_indices = torch.argmin(distances, dim=1) # 
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)   
class SNN_VQVAE_uni(nn.Module):

    """VQ-VAE"""
    
    def __init__(self, in_dim, embedding_dim, num_embeddings, data_variance, 
        commitment_cost=0.25):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.data_variance = data_variance
        
        self.encoder = Encoder(in_dim, embedding_dim)
        self.vq_layer = VectorQuantizer_uni(embedding_dim, num_embeddings, commitment_cost)
        self.decoder = Decoder(in_dim, embedding_dim)
        # TODO:
        self.memout = MembraneOutputLayer()

    def forward(self, x,image):
        # x: [t, B, C, H, W]
        z = self.encoder(x)
        # print(z.shape)
        if not self.training:
            e,enco = self.vq_layer(z)
            x_recon = self.decoder(e)
            x_recon = torch.tanh(self.memout(x_recon))
            return e, x_recon,enco
        
        e, e_q_loss,FID_loss = self.vq_layer(z)
        x_recon = self.decoder(e)
        x_recon = torch.tanh(self.memout(x_recon))
        
        real_recon_loss = F.mse_loss(x_recon, image)
        recon_loss = real_recon_loss / self.data_variance
        
        return e_q_loss+FID_loss,recon_loss,real_recon_loss
    

    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim 
        self.num_embeddings = num_embeddings 
        self.commitment_cost = commitment_cost 
        self.memout = MembraneOutputLayer()
        self.num_step = 16
        self.psp = PSP()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
        
    def poisson(self,x):
        return x/128
    
    def forward(self, x):
        # (T,N,C,H,W)->(N,C,H,W) 16 128 16 7 7
        x_memout = (1-self.alpha)*self.memout(x) + self.alpha*torch.sum(x,dim=0)/self.num_step
        # [128, 16, 7, 7] -> [128, 7, 7,16]
        x_memout = x_memout.permute(0, 2, 3, 1).contiguous()
        # [128, 7, 7,16] -> [6272, 16]
        flat_x = x_memout.reshape(-1, self.embedding_dim)
        # encoding_indices是离散值，形状latent-dim
        encoding_indices = self.get_code_indices(flat_x)

        quantized = self.quantize(encoding_indices)

        quantized = quantized.view_as(x_memout) # [128, 7, 7, 16]
        
        if not self.training:
            quantized = quantized.permute(0, 3, 1, 2).contiguous() #[128, 16, 7, 7]
            quantized = torch.unsqueeze(quantized, dim=0) #[1, 128, 16, 7, 7]
            quantized = quantized.repeat(16, 1, 1, 1, 1) #[16, 128, 16, 7, 7]
            quantized = self.poisson(quantized) # [16, 128, 16, 7, 7]
            return quantized,encoding_indices
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x_memout.detach())

        # commitment loss
        e_latent_loss = F.mse_loss(x_memout, quantized.detach())

        loss_1 = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator 
        # forward quantized = quantized
        # backward quantized = x
        quantized = x_memout + (quantized - x_memout).detach()
        # [128, 7, 7,16]->[128,16,7,7]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # FIXME:
        quantized = torch.unsqueeze(quantized, dim=0) # [1,128,16,7,7] 
        quantized = quantized.repeat(16, 1, 1, 1, 1) # [16,128,16,7,7]
        quantized = self.poisson(quantized) # [16,128,16,7,7]

        q_latent_loss_2 = torch.mean((self.psp(quantized)-self.psp(x.detach()))**2)
        e_latent_loss_2 = torch.mean((self.psp(quantized.detach())-self.psp(x))**2)
        loss_2 =  q_latent_loss_2 + self.commitment_cost * e_latent_loss_2
        
        return quantized, loss_1 #+loss_2

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.weight.t())
        ) 
        encoding_indices = torch.argmin(distances, dim=1) 
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)   
