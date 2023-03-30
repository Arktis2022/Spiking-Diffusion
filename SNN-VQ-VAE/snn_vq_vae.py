import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import sys
from fsvae_models.snn_layers import *

# vq层，用于实现vq嵌入
class VectorQuantizer(nn.Module):
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
        self.embedding_dim = embedding_dim # 向量维度16
        self.num_embeddings = num_embeddings # 量化空间中 向量个数-128
        self.commitment_cost = commitment_cost # β 损失最后一项 约束项 0.25
        self.memout = MembraneOutputLayer()
        self.num_step = 16
        # 用于将离散的输入（如单词或类别）映射到低维空间中的连续向量表示
        # 输入是num_embeddings输出是embedding_dim
        # 意思是，例如输入形状为[num_embeddings = 10],则输出形状为[10,embedding_dim = 128]
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

        self.poisson = tdConv(16,
                        out_channels=16,
                        kernel_size=1,
                        stride = 1,
                        padding = 0, 
                        bias=True,
                        bn=tdBatchNorm(16),
                        spike=LIFSpike(),
                        is_first_conv=True)
        # 因此其参数形状即为[10 x 128]，参数构成了所需要的向量空间，非常巧妙，并不用作直接处理数据

    def forward(self, x):
        # (N,C,H,W,T)->(N,C,H,W)
        
        #x_memout = torch.sum(x,dim=-1)/self.num_step
        x_memout = self.memout(x)
        # channel 挪到最后
        # # [128, 16, 7, 7] -> [128, 7, 7,16]
        x_memout = x_memout.permute(0, 2, 3, 1).contiguous()

        #x = x.permute(0,2,3,1,4).contiguous()
        
        # [128, 7, 7,16] -> [6272, 16]
        flat_x = x_memout.reshape(-1, self.embedding_dim)

        # 将flat_x去和潜在空间中存的向量去比较，得到编码索引，形状为[BHW]，每个维度获得了一个索引
        # encoding_indices是离散值，形状latent-dim
        encoding_indices = self.get_code_indices(flat_x)

        # 输出形状[6272]——128个图片，每个图片有7x7=49个维度，每个维度的向量长度为16
        # 因此该输出将每个维度的长度为16的向量编码为单个索引

        quantized = self.quantize(encoding_indices)
        #再将索引转为编码，输出形状[6272,16]

        quantized = quantized.view_as(x_memout) # [128, 7, 7,16]
        


        if not self.training:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            quantized = torch.unsqueeze(quantized, dim=4)
            quantized = quantized.repeat(1, 1, 1, 1, 16)
            quantized = self.poisson(quantized)
            return quantized
        

        '''
        解释这两个损失：
        q_latent_loss——用于将嵌入向量像编码向量x靠拢
        e_latent_loss——用于将编码向量向嵌入向量靠拢
        '''
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x_memout.detach())
        # 这个损失用于对比嵌入向量与输入向量的相似度

        # commitment loss
        e_latent_loss = F.mse_loss(x_memout, quantized.detach())
        
        # 两个loss权重不一样
        loss_1 = q_latent_loss + self.commitment_cost * e_latent_loss


        # Straight Through Estimator
        quantized = x_memout + (quantized - x_memout).detach()
        # [128, 7, 7,16]->[128,16,7,7]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        '''
        part-2
        '''
        # [128,16,7,7]->[128,16,7,7,16]
        quantized = torch.unsqueeze(quantized, dim=4)
        quantized = quantized.repeat(1, 1, 1, 1, 16)

        quantized = self.poisson(quantized) # [128,16,7,7,16]
        q_latent_loss_2 = F.mse_loss(quantized, x.detach())
        e_latent_loss_2 = F.mse_loss(x, quantized.detach())
        quantized = x + (quantized - x).detach()
        loss_2 =  q_latent_loss_2 + self.commitment_cost * e_latent_loss
        
        return quantized, loss_1+loss_2
    
    # 计算l2距离
    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.weight.t())
        ) # [N, M]——N个维度，每个维度M个距离
        encoding_indices = torch.argmin(distances, dim=1) # [N,] # 找出每个维度最短的距离
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)      

class Encoder(nn.Module):
    """Encoder of VQ-VAE"""
    
    def __init__(self, in_dim=3, latent_dim=16):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        
        is_first_conv = True
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_dim, 1),
        )

        self.snn_convs = nn.Sequential(
            tdConv(in_dim,
                        out_channels=32,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(32),
                        spike=LIFSpike(),
                        is_first_conv=is_first_conv),

            tdConv(32,
                        out_channels=64,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(64),
                        spike=LIFSpike(),
                        is_first_conv=is_first_conv),

            tdConv(64,
                        out_channels=latent_dim,
                        kernel_size=1, 
                        bias=True,
                        bn=tdBatchNorm(latent_dim),
                        spike=LIFSpike(),
                        is_first_conv=is_first_conv)
            )
        
    def forward(self, x):
        return self.snn_convs(x)
    

class Decoder(nn.Module):
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
        
        self.snn_convs = nn.Sequential(
            tdConvTranspose(latent_dim,
                                    64,
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(64),
                                    spike=LIFSpike()),
            tdConvTranspose(64,
                                    32,
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(32),
                                    spike=LIFSpike()),
            tdConvTranspose(32,
                                    out_dim,
                                    kernel_size=3,
                                    padding=1,
                                    bias=True,
                                    bn=None,
                                    spike=None)
        )
    def forward(self, x):
        return self.snn_convs(x)

class VQVAE(nn.Module):

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
        self.memout = MembraneOutputLayer()
        
    def forward(self, x,image):

        z = self.encoder(x)

        if not self.training:
            e = self.vq_layer(z)
            x_recon = self.decoder(e)
            x_recon = self.memout(x_recon)
            return e, x_recon
        
        e, e_q_loss = self.vq_layer(z)
        x_recon = self.decoder(e)
        x_recon = self.memout(x_recon)
        tensor = x_recon
        recon_loss = F.mse_loss(x_recon, image) / self.data_variance
        
        return e_q_loss,recon_loss    

if __name__ == '__main__':
    batch_size = 128
    embedding_dim = 16
    num_embeddings = 128

    transform=transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset1 = datasets.MNIST('/data/liumingxuan/VQ-VAE/mnist/', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('/data/liumingxuan/VQ-VAE/mnist/', train=False,
                        transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)

    # compute the variance of the whole training set to normalise the Mean Squared Error below.
    train_images = []
    for images, labels in train_loader:
        train_images.append(images)
    train_images = torch.cat(train_images, dim=0)
    train_data_variance = torch.var(train_images)

    model = VQVAE(1, embedding_dim, num_embeddings, train_data_variance)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train VQ-VAE
    epochs = 100
    print_freq = 20

    for epoch in range(epochs):
        model.train()

        print("Start training epoch {}".format(epoch,))
        for i, (images, labels) in enumerate(train_loader):
            images = images - 0.5 # normalize to [-0.5, 0.5]
            images = images.cuda()
            images_spike = images.unsqueeze(-1).repeat(1, 1, 1, 1, 16)
            loss_eq,loss_rec = model(images_spike,images)
            optimizer.zero_grad()
            (loss_eq+loss_rec).backward()
            optimizer.step()
            if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
                print("[{}/{}][{}/{}]: loss {:.3f} loss_eq {:.3f} loss_rec {:.3f}".format(epoch,epochs,i,len(train_loader),(loss_eq+loss_rec).item(),float(loss_eq),float(loss_rec)))
            
        # reconstructe images
        test_loader_iter = iter(test_loader)
        images, labels = next(test_loader_iter)

        n_samples = 32
        images = images[:n_samples]

        model.eval()

        norm_images = (images - 0.5).cuda()
        with torch.inference_mode():
            images_spike = norm_images.unsqueeze(-1).repeat(1, 1, 1, 1, 16)
            e, recon_images = model(images_spike,norm_images)

        recon_images = np.array(np.clip((recon_images + 0.5).cpu().numpy(), 0., 1.) * 255, dtype=np.uint8)
        ori_images = np.array(images.numpy() * 255, dtype=np.uint8)

        recon_images = recon_images.reshape(4, 8, 28, 28)
        ori_images = ori_images.reshape(4, 8, 28, 28)

        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        gs = fig.add_gridspec(8, 8)
        for n_row in range(4):
            for n_col in range(8):
                f_ax = fig.add_subplot(gs[n_row * 2, n_col])
                f_ax.imshow(ori_images[n_row, n_col], cmap="gray")
                f_ax.axis("off")
                f_ax = fig.add_subplot(gs[n_row * 2 + 1, n_col])
                f_ax.imshow(recon_images[n_row, n_col], cmap="gray")
                f_ax.axis("off")
        plt.savefig("./result/epoch="+str(epoch)+"_test.png")




