import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 该文件对snn进行建模

dt = 5
a = 0.25
aa = 0.5  
Vth = 0.2
tau = 0.25

# torch.autograd.Function 是 PyTorch 中用来定义自定义自动求导函数的模板类。
# 这些函数接受输入的张量并返回输出张量，同时计算输入张量相对于某些变量的梯度。这些函数通常是非线性的，并且不像简单的数学函数一样可以使用符号微分来计算梯度。
# 该部分用于实现尖峰发射函数的梯度近似
class SpikeAct(torch.autograd.Function):
    """ 
        利用梯度近似实现尖峰激活函数
    """
    @staticmethod
    def forward(ctx, input):
        
        # 在forward方法中，首先通过ctx.save_for_backward保存了输入张量input，以便在反向传播时使用
        ctx.save_for_backward(input)
        
        # if input = u > Vth then output = 1
        # torch.gt用于比较input和Vth的大小，返回的是true or false，也就是0或1
        output = torch.gt(input, Vth) 
        
        # 最后将逻辑张量转为浮点张量
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        
        #在backward方法中，首先通过ctx.saved_tensors获取之前保存的输入张量input
        input, = ctx.saved_tensors 
        
        # 先保存传入的梯度
        grad_input = grad_output.clone()
        
        # 给出尖峰发射函数的近似梯度df/dt
        hu = abs(input) < aa
        hu = hu.float() / (2 * aa)
        return grad_input * hu

class LIFSpike(nn.Module):
    """
        基于LIF模块生成峰值。它可以被视为一种激活功能，其使用类似于ReLU。输入张量需要有一个额外的时间维度，在这种情况下是在数据的最后一个维度上。
    """
    def __init__(self):
        super(LIFSpike, self).__init__()

    def forward(self, x):
        # 现在执行前向传播
        
        # 时间步数是x.shape[-1]，即输入的最后一个维度
        nsteps = x.shape[-1]
        
        # 膜电位u，首先进行初始化
        u   = torch.zeros(x.shape[:-1] , device=x.device)
        
        # 输出脉冲，初始化
        out = torch.zeros(x.shape, device=x.device)
        
        # 对于所有时间步长的输入
        for step in range(nsteps):
            
            # 根据这个函数更新u，并给出输出，这个部分不难
            # 输入是：
            # 之前的膜电位u
            # out[..., max(step-1, 0)]是什么？哦应该是看上一步有没有发出脉冲，有的话得膜电位归0
            # x[..., step]是当前的输入
            u, out[..., step] = self.state_update(u, out[..., max(step-1, 0)], x[..., step])
        return out
    
    # 定义状态更新函数
    def state_update(self, u_t_n1, o_t_n1, W_mul_o_t1_n, tau=tau):
        # 更新膜电位
        u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
        # 根据膜电位得到输出脉冲
        o_t1_n1 = SpikeAct.apply(u_t1_n1)
        return u_t1_n1, o_t1_n1  

# td线性化层，这是集成了bn层操作吗
class tdLinear(nn.Linear):
    def __init__(self, in_features,out_features,bias=True,bn=None,spike=None):
        
        # 输入输出都是一个维度
        assert type(in_features) == int, 'inFeatures should not be more than 1 dimesnion. It was: {}'.format(in_features.shape)
        assert type(out_features) == int, 'outFeatures should not be more than 1 dimesnion. It was: {}'.format(out_features.shape)
        
        super(tdLinear, self).__init__(in_features, out_features, bias=bias)
        
        self.bn = bn
        self.spike = spike
        
    
    # 输入形状N,C,T，其中N是batch_size，C是特征向量，T是时间步长
    def forward(self, x):
        """
        x : (N,C,T)
        """        
        # 这一步就是正常的全连接层操作
        x = x.transpose(1, 2) # (N, T, C)
        y = F.linear(x, self.weight, self.bias)
        y = y.transpose(1, 2)# (N, C, T)
        
        # 如果要进行依赖于阈值的bn
        if self.bn is not None:
            
            # y的形状为（N,C,T）
            # 通过这种方法为y增加两个维度
            # 目的是为了适配bn的通用方法吧，bn是针对（N，C,W,H,T）这种形状执行的
            y = y[:,:,None,None,:]
            
            # 对y执行批归一化
            y = self.bn(y)
            
            # 输出之后的y，形状依然是（N,C,T）
            y = y[:,:,0,0,:]
            
        # 如果要输出脉冲
        if self.spike is not None:
            y = self.spike(y)
        return y

# 带有tdbn的卷积块
class tdConv(nn.Conv3d):
    def __init__(self, 
                in_channels, 
                out_channels,  
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                bn=None,
                spike=None,
                is_first_conv=False):

        # 卷积核设置
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # 步长设置
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding设置
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        super(tdConv, self).__init__(in_channels, out_channels, kernel, stride, padding, dilation, groups,
                                        bias=bias)
        self.bn = bn
        self.spike = spike
        self.is_first_conv = is_first_conv
    
    # 前向传播
    def forward(self, x):
        # 居然是用3D卷积，感觉不太符合SNN的实际运行
        x = F.conv3d(x, self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
        
        # 比起普通操作就多了一个自定义的bn层
        if self.bn is not None:
            x = self.bn(x)
        if self.spike is not None:
            x = self.spike(x)
        return x
        

# 有tdbn的反卷积块
class tdConvTranspose(nn.ConvTranspose3d):
    def __init__(self, 
                in_channels, 
                out_channels,  
                kernel_size,
                stride=1,
                padding=0,
                output_padding=0,
                dilation=1,
                groups=1,
                bias=True,
                bn=None,
                spike=None):

        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))


        # output padding
        if type(output_padding) == int:
            output_padding = (output_padding, output_padding, 0)
        elif len(output_padding) == 2:
            output_padding = (output_padding[0], output_padding[1], 0)
        else:
            raise Exception('output_padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        super().__init__(in_channels, out_channels, kernel, stride, padding, output_padding, groups,
                                        bias=bias, dilation=dilation)

        self.bn = bn
        self.spike = spike

    def forward(self, x):
        # 三维卷积
        # 接受一个形状为(N,C,W,H,T)的张量x作为输入，其中N是批量大小，C是通道数，W和H是空间维度大小，T是时间步长
        # 对C,W,H,T进行卷积操作
        x = F.conv_transpose3d(x, self.weight, self.bias,
                        self.stride, self.padding, 
                        self.output_padding, self.groups, self.dilation)

        if self.bn is not None:
            x = self.bn(x)
        if self.spike is not None:
            x = self.spike(x)
        return x

# 进行tdBN
class tdBatchNorm(nn.BatchNorm2d):
    """
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4])
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * Vth * (input - mean[None, :, None, None, None]) / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]
        
        return input

# 模拟突触后电位，应该是最后一层
class PSP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tau_s = 2

    def forward(self, inputs):
        """
        inputs: (N, C, T)
        """
        # 输入是N C T
        syns = None
        syn = 0
        n_steps = inputs.shape[-1]
        
        for t in range(n_steps):
            syn = syn + (inputs[...,t] - syn) / self.tau_s
            if syns is None:
                syns = syn.unsqueeze(-1)
            else:
                syns = torch.cat([syns, syn.unsqueeze(-1)], dim=-1)

        return syns

class MembraneOutputLayer(nn.Module):
    """
    outputs the last time membrane potential of the LIF neuron with V_th=infty
    输出LIF神经元的最后一次膜电位，V_th=infty
    """
    def __init__(self) -> None:
        super().__init__()
        n_steps = 16

        arr = torch.arange(n_steps-1,-1,-1)
        self.register_buffer("coef", torch.pow(0.8, arr)[None,None,None,None,:]) # (1,1,1,1,T)

    def forward(self, x):
        """
        x : (N,C,H,W,T)
        """
        out = torch.sum(x*self.coef, dim=-1)
        return out
    
class MembraneOutputLayer_for_dsvdd(nn.Module):
    """
    outputs the last time membrane potential of the LIF neuron with V_th=infty
    输出LIF神经元的最后一次膜电位，V_th=infty
    """
    def __init__(self) -> None:
        super().__init__()
        n_steps = 16

        arr = torch.arange(n_steps-1,-1,-1)
        self.register_buffer("coef", torch.pow(0.8, arr)[None,None,:]) # (1,1,T)

    def forward(self, x):
        """
        x : (N,C,T)
        """
        out = torch.sum(x*self.coef, dim=-1)
        return out