
import torch
import torch.nn as nn
from .snn_layers import *
from .fsvae_prior import *
from .fsvae_posterior import *
import torch.nn.functional as F

import global_v as glv


# 重点内容，FSVAE
class FSVAE(nn.Module):
    def __init__(self,class_num):
        super().__init__()
        
        # 输入通道基本上是1 二进制序列
        in_channels = glv.network_config['in_channels']
        
        # 中间层z的维度——128  隐空间Z维度
        latent_dim = glv.network_config['latent_dim']

        self.class_num = class_num
        self.latent_dim = latent_dim
        
        # 时间步长——只需要16即可
        self.n_steps = glv.network_config['n_steps']
        
        # k是什么？  采样前维度: latent_dim*k
        self.k = glv.network_config['k']
        
        # 中间层维度
        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # 球心
        self.c = nn.Parameter(torch.randn(size=(class_num, latent_dim)))
        self.r = nn.Parameter(torch.randn(1))
        # 编码器
        modules = []
        is_first_conv = True
        
        # 对于隐藏层维度
        for h_dim in hidden_dims:
            
            # 设置了有tdbn的卷积层
            modules.append(
                tdConv(in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike(),
                        is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False
        
        # 编码器即为以上
        self.encoder = nn.Sequential(*modules)
        
        # 在隐藏层往前一层，这一层作用是卷积转线性吧
        self.before_latent_layer = tdLinear(hidden_dims[-1]*4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())
        
        # 这里是作者搭建的先验生成器
        self.prior = PriorBernoulliSTBP(self.k)
        
        # 这里是作者搭建的后验生成器
        self.posterior = PosteriorBernoulliSTBP(self.k)

        # 建造解码器
        modules = []
        
        # 解码器输入构造
        self.decoder_input = tdLinear(latent_dim, 
                                        hidden_dims[-1] * 4, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * 4),
                                        spike=LIFSpike())
        
        # 将编码器的中间dim反转
        hidden_dims.reverse()
        
        # 反转后少个一层
        for i in range(len(hidden_dims) - 1):
            
            # 反卷积层
            modules.append(
                    tdConvTranspose(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)
        
        # 最后一层网络
        self.final_layer = nn.Sequential(
                            tdConvTranspose(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=True,
                                            bn=tdBatchNorm(hidden_dims[-1]),
                                            spike=LIFSpike()),
                            tdConvTranspose(hidden_dims[-1], 
                                            out_channels=glv.network_config['in_channels'],
                                            kernel_size=3, 
                                            padding=1,
                                            bias=True,
                                            bn=None,
                                            spike=None)
        )

        self.p = 0
        
        # 输出层
        self.membrane_output_layer = MembraneOutputLayer()
        
        # 这里是模拟突触后电位
        self.psp = PSP()
        
        
        # 输入形状（B,C*H*W,T）
        # 输出形状为为(B,C,T)
        self.lif_for_dsvdd = tdLinear(1024,latent_dim,bias=True,bn=tdBatchNorm(latent_dim),spike=LIFSpike())
        self.lif_for_clf = tdLinear(latent_dim,class_num,bias=True,bn=tdBatchNorm(class_num),spike=LIFSpike())
        # dsvdd的输出层
        self.output_for_dsvdd = MembraneOutputLayer_for_dsvdd()

        # 增加一个分类器
        self.output_for_clf = MembraneOutputLayer_for_dsvdd()

    def update_c(self,c=None):
        if c!=None:
            self.c = torch.nn.Parameter(c)
        return self.c

    def update_r(self,r=None):
        if r!= None:
            self.r = torch.nn.Parameter(r)
        return self.r

    # 前向传播
    def forward(self, x, scheduled=False):
        sampled_z, q_z, p_z,dsvdd_z,clf = self.encode(x, scheduled)

        x_recon = self.decode(sampled_z)
        return x_recon, q_z, p_z, sampled_z,dsvdd_z,clf
    
    # 编码器
    def encode(self, x, scheduled=False):
        
        # 输入x经过编码得到形状为(N,C,H,W,T)的东西
        x = self.encoder(x) # (N,C,H,W,T)
        
        # 将x线性化，老生常谈了
        x = torch.flatten(x, start_dim=1, end_dim=3) # (N,C*H*W,T)
        
        # 给出dsvdd需要的特征z
        dsvdd_z_f = (self.lif_for_dsvdd(x))
        dsvdd_z = self.output_for_dsvdd(dsvdd_z_f) # (batch_size,latent_dim)
        clf = self.output_for_clf(self.lif_for_clf(dsvdd_z_f)) # (batch_size,class_num)
        
        # 将线性化的x再全连接一下
        latent_x = self.before_latent_layer(x) # (N,latent_dim,T)
        
        
        # 后验z生成 考虑输入x
        # sampled_z是啥，应该是为了生成例子用的一些z吧
        # 编码后得到的z并不是需要的Z 还需要经过伯努利采样
        # 这里采样为什么不会出现反向传播的问题？
        sampled_z, q_z = self.posterior(latent_x) # sampled_z:(B,C,1,1,T), q_z:(B,C,k,T)
        
        # 先验z生成，输入是从后验z中的采样
        p_z = self.prior(sampled_z, scheduled, self.p)
        return sampled_z, q_z, p_z,dsvdd_z,clf

    # 解码器，根据输入的z解码
    def decode(self, z):
        result = self.decoder_input(z) # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, self.n_steps) # (N,C,H,W,T)
        result = self.decoder(result)# (N,C,H,W,T)
        result = self.final_layer(result)# (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result))        
        return out
    
    # 采样，根据采样输出结果
    # 这个函数用到了嘛？？？？
    def sample(self, batch_size=64):
        sampled_z = self.prior.sample(batch_size)
        sampled_x = self.decode(sampled_z)
        return sampled_x, sampled_z
    
    # 定义损失函数，mmd损失
    # 除了输入图像与输出图像的差异 重构损失
    # 还有q_z与p_z的差异 正则化损失
    def loss_function_mmd(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        q_z_ber = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        p_z_ber = torch.mean(p_z, dim=2) # (N, latent_dim, T)

        #kld_loss = torch.mean((q_z_ber - p_z_ber)**2)
        mmd_loss = torch.mean((self.psp(q_z_ber)-self.psp(p_z_ber))**2)
        loss = recons_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Distance_Loss': mmd_loss}
    
    # 定义损失函数，kld损失
    def loss_function_kld(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        prob_q = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        prob_p = torch.mean(p_z, dim=2) # (N, latent_dim, T)
        
        kld_loss = prob_q * torch.log((prob_q+1e-2)/(prob_p+1e-2)) + (1-prob_q)*torch.log((1-prob_q+1e-2)/(1-prob_p+1e-2))
        kld_loss = torch.mean(torch.sum(kld_loss, dim=(1,2)))

        loss = recons_loss + 1e-4 * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Distance_Loss': kld_loss}
    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4,4)

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p-init_p) * epoch / max_epoch + init_p
        

class FSVAELarge(FSVAE):
    def __init__(self):
        super(FSVAE, self).__init__()
        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1]*4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        self.prior = PriorBernoulliSTBP(self.k)
        
        self.posterior = PosteriorBernoulliSTBP(self.k)
        
        # Build Decoder
        modules = []
        
        self.decoder_input = tdLinear(latent_dim, 
                                        hidden_dims[-1] * 4, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * 4),
                                        spike=LIFSpike())
        
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                    tdConvTranspose(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            tdConvTranspose(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=True,
                                            bn=tdBatchNorm(hidden_dims[-1]),
                                            spike=LIFSpike()),
                            tdConvTranspose(hidden_dims[-1], 
                                            out_channels=glv.network_config['in_channels'],
                                            kernel_size=3, 
                                            padding=1,
                                            bias=True,
                                            bn=None,
                                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()