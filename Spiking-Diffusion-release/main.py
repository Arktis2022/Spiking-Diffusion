import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
import sys
import tqdm
import argparse
import random
from snn_model.snn_layers import *
from snn_model.vae_model import *
from snn_model.vq_diffusion import *
from load_dataset_snn import *
import metric.pytorch_ssim

from metric.IS_score import *
from metric.Fid_score import *
from torchmetrics.image.kid import KernelInceptionDistance
from syops import get_model_complexity_info
#from sklearn.mixture import GaussianMixture
#import warnings
#warnings.filterwarnings("ignore")
'''
指标：
1. 测试集的重建损失——mse损失，ssim损失
2. 推断单个图像需要多少浮点加法和乘法
3. 采样：
    + IS分数
    + FID分数
    + MMD分数
'''
# 随机数
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store', dest='checkpoint',
                        help='The path of checkpoint, if use checkpoint')
    parser.add_argument('--dataset_name', type=str,default='MNIST')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--model',type=str,default='vq-vae')
    parser.add_argument('--data_path',type=str,default='/data/liumingxuan/SNN-VAE-DMSVDD/datasets/Datasets')
    parser.add_argument('--sample_model',type=str,default='pixelsnn')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--metric',type=str,default=None)
    parser.add_argument('--ready',type=str,default=None)
    parser.add_argument('--mask',type=str,default='codebook_size')
    parser.add_argument('--codebook_size',type=int,default=128)
    args = parser.parse_args()

    setup_seed(args.seed)
    if not os.path.exists("./result/"+args.dataset_name+'/'+args.model):
            os.makedirs("./result/"+args.dataset_name+'/'+args.model)
    save_path = "./result/"+args.dataset_name+'/'+args.model

    # 固定的超参数
    batch_size_1 = 32
    batch_size_2 = 32
    embedding_dim = 16
    num_embeddings = args.codebook_size
    if args.dataset_name == 'MNIST':
        train_loader,test_loader = load_mnist(data_path = args.data_path,batch_size=batch_size_1)
        train_loader_2,test_loader_2 = load_mnist(data_path = args.data_path,batch_size=batch_size_2)
        print("load data: MNIST!")
    elif args.dataset_name == 'KMNIST':
        train_loader,test_loader = load_KMNIST(data_path = args.data_path,batch_size=batch_size_1)
        train_loader_2,test_loader_2 = load_KMNIST(data_path = args.data_path,batch_size=batch_size_2)
        print("load data: KMNIST!")
    elif args.dataset_name == 'FMNIST':
        train_loader,test_loader = load_fashionmnist(data_path = args.data_path,batch_size=batch_size_1)
        train_loader_2,test_loader_2 = load_fashionmnist(data_path = args.data_path,batch_size=batch_size_2)
        print("load data: FMNIST!")
    elif args.dataset_name == 'Letters':
        train_loader,test_loader = load_MNIST_Letters(data_path = args.data_path,batch_size=batch_size_1)
        train_loader_2,test_loader_2 = load_MNIST_Letters(data_path = args.data_path,batch_size=batch_size_2)
        print("load data: Letters!")


    
    # compute the variance of the whole training set to normalise the Mean Squared Error below.
    train_images = []
    for images, labels in train_loader:
        train_images.append(images)
    train_images = torch.cat(train_images, dim=0)
    train_data_variance = torch.var(train_images)

    if args.model == 'snn-vq-vae':
        model = SNN_VQVAE(1, embedding_dim, num_embeddings, train_data_variance)
        functional.set_step_mode(net = model, step_mode = 'm')
    elif args.model == 'snn-vq-vae-uni':
        model = SNN_VQVAE_uni(1, embedding_dim, num_embeddings, train_data_variance)
        functional.set_step_mode(net = model, step_mode = 'm')
    elif args.model == 'snn-vae':
        model = SNN_VAE()
        functional.set_step_mode(net = model, step_mode = 'm')
    elif args.model == 'vq-vae':
        model = VQVAE(1, embedding_dim, num_embeddings, train_data_variance)
    model = model.cuda(0)
    print("The model is ready!")


    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=1e-3, 
                                betas=(0.9, 0.999), 
                                weight_decay=0.001)

    # train VQ-VAE
    epochs = args.epochs
    print_freq = 20
    
    if args.checkpoint==None:
        if args.ready==None:
            for epoch in range(epochs):
                model.train()
                
                print("Start training epoch {}".format(epoch,))
                if args.model=='vae':
                    model.update_p(epoch,epochs)
                for i, (images, labels) in enumerate(train_loader):
                    images = images - 0.5 # normalize to [-0.5, 0.5]
                    images = images.cuda(0)
                    images_spike = images.unsqueeze(0).repeat(16, 1, 1, 1, 1)
                    if args.model=='snn-vae':
                        loss_eq,loss_rec = model(images_spike,images)
                    elif args.model=='snn-vq-vae' or args.model=='snn-vq-vae-uni':
                        loss_eq,loss_rec,real_loss_rec = model(images_spike,images)
                    elif args.model=='vq-vae':
                        loss_eq,loss_rec, real_loss_rec= model(images)
                    optimizer.zero_grad()
                    (loss_eq+loss_rec).backward()
                    optimizer.step()

                    if args.model!='vq-vae':
                        functional.reset_net(model)

                    if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
                        if args.model=='snn-vae':
                            print("[{}/{}][{}/{}]: loss {:.3f} loss_eq {:.3f} loss_rec {:.3f}".format(epoch,epochs,i,len(train_loader),(loss_eq+loss_rec).item(),float(loss_eq),float(loss_rec)))
                            #break
                        elif args.model=='snn-vq-vae' or args.model=='vq-vae' or args.model=='snn-vq-vae-uni':
                            print("[{}/{}][{}/{}]: loss {:.3f} loss_eq {:.3f} loss_rec {:.3f}".format(epoch,epochs,i,len(train_loader),(loss_eq+loss_rec).item(),float(loss_eq),float(real_loss_rec)))
                    #break
                # reconstructe images
                test_loader_iter = iter(test_loader)
                images, labels = next(test_loader_iter)

                n_samples = 32
                images = images[:n_samples]

                model.eval()

                norm_images = (images - 0.5).cuda(0)
                with torch.inference_mode():
                    if args.model=='vq-vae':
                        e, recon_images = model(norm_images)
                    elif args.model=='snn-vae':
                        images_spike = norm_images.unsqueeze(0).repeat(16, 1, 1, 1, 1)
                        e, recon_images = model(images_spike,norm_images)
                        functional.reset_net(model)
                    elif args.model== ('snn-vq-vae'):
                        images_spike = norm_images.unsqueeze(0).repeat(16, 1, 1, 1, 1)
                        e, recon_images,_ = model(images_spike,norm_images)
                        functional.reset_net(model)
                    elif args.model=="snn-vq-vae-uni":
                        images_spike = norm_images.unsqueeze(0).repeat(16, 1, 1, 1, 1)
                        e, recon_images,_ = model(images_spike,norm_images)
                        functional.reset_net(model)

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
                
                plt.savefig("./result/"+args.dataset_name+'/'+args.model+"/epoch="+str(epoch)+"_test.png")
        
                torch.save(model.state_dict(), save_path+'/model.pth')

        # train diffusion
        if args.model=='snn-vq-vae' or args.model=='vq-vae' or args.model=='snn-vq-vae-uni':
            if args.ready==None:
                model.load_state_dict(torch.load(save_path+'/model.pth'))
            elif args.ready!=None:
                model.load_state_dict(torch.load(args.ready))
            
            train_indices = get_data_for_diff(train_loader_2,model)
            print(len(train_indices))
            print(train_indices[0].shape)
            print(train_indices[0][0])
            if args.mask=='codebook_size':
                mask_id = num_embeddings
            elif args.mask=='max':
                most_common_value, count = torch.mode(torch.flatten(train_indices[0]))
                mask_id = most_common_value
            elif args.mask=='min':
                values, counts = torch.unique(torch.flatten(train_indices[0]), return_counts=True)
                least_common_value = values[torch.min(counts)].item()
                count = torch.min(counts).item()
                mask_id = count

            print("mask_id = ",mask_id)
            print('data for train diffusion is ready!')

            denoise_fn = DummyModel(1,num_embeddings).cuda(0)
            functional.set_step_mode(net = denoise_fn, step_mode = 'm')
            abdiff = AbsorbingDiffusion(denoise_fn, mask_id=mask_id)

            epochs = args.epochs*2
            if not os.path.exists("./result/"+args.dataset_name+'/'+args.model+'/diff_result'):
                os.makedirs("./result/"+args.dataset_name+'/'+args.model+'/diff_result')
            save_path = "./result/"+args.dataset_name+'/'+args.model+'/diff_result'

            #optimizer = torch.optim.Adam(denoise_fn.parameters(), lr=1e-3)
            optimizer = torch.optim.AdamW(denoise_fn.parameters(), 
                                lr=1e-3, 
                                betas=(0.9, 0.999), 
                                weight_decay=0.001)

            for epoch in range(epochs):
                
                denoise_fn.train()
                for batch_idx, (indices) in enumerate(train_indices):
                    indices = indices.float().cuda(0)
                    indices = indices.unsqueeze(dim=1)
                    loss = abdiff.train_iter(indices)
                    loss = loss['loss']
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    functional.reset_net(net = denoise_fn)
                    print("[{}/{}][{}/{}]: loss {:.3f} ".format(epoch,epochs,batch_idx,len(train_loader_2),(loss).item()))
                    #break
                if epoch%10 == 0:
                
                    denoise_fn.eval() 
                    sample_list = []
                    for i in range(2):
                        sample = (abdiff.sample(temp = 0.65, sample_steps=49)).reshape(16,7,7)
                        sample_list.append(sample)
                    sample = torch.cat(sample_list,dim=0)
                    with torch.inference_mode():
                        z = model.vq_layer.quantize(sample.cuda(0))

                        z = z.permute(0, 3, 1, 2).contiguous()
                        quantized = torch.unsqueeze(z, dim=0)
                        quantized = quantized.repeat(16, 1, 1, 1, 1)
                        quantized = model.vq_layer.poisson(quantized)
                        # torch.Size([128, 16, 7, 7, 16])

                        pred = model.decoder(quantized)
                        pred = torch.tanh(model.memout(pred))

                    generated_samples = np.array(np.clip((pred + 0.5).cpu().numpy(), 0., 1.) * 255, dtype=np.uint8)
                    generated_samples = generated_samples.reshape(4, 8, 28, 28)

                    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
                    gs = fig.add_gridspec(4, 8)
                    for n_row in range(4):
                        for n_col in range(8):
                            f_ax = fig.add_subplot(gs[n_row, n_col])
                            f_ax.imshow(generated_samples[n_row, n_col], cmap="gray")
                            f_ax.axis("off")
                    plt.savefig(save_path+"/epoch="+str(epoch)+"_test.png")
                    torch.save(denoise_fn.state_dict(), save_path+'/diff_model.pth')

    else:
        model.load_state_dict(torch.load(args.checkpoint))
        functional.set_step_mode(net = model, step_mode = 'm')
        if args.model==('snn-vq-vae'or'snn-vq-vae-uni') or args.model=='vq-vae':
            mask_id=num_embeddings
            denoise_fn = DummyModel(1,num_embeddings).cuda(0)
            functional.set_step_mode(net = denoise_fn, step_mode = 'm')
            denoise_fn.load_state_dict(torch.load(args.checkpoint[:-10]+'/diff_result/diff_model.pth'))
            abdiff = AbsorbingDiffusion(denoise_fn, mask_id=mask_id)



    # 测试模型性能
    model.eval()

    # 重建误差测试
    loss_mse = []
    loss_ssim = []
    for i, (images, labels) in enumerate(test_loader_2):
        norm_images = (images - 0.5).cuda(0)
        with torch.inference_mode():
            images_spike = norm_images.unsqueeze(0).repeat(16, 1, 1, 1, 1)
            if args.model=='vq-vae':
                e, recon_images = model(norm_images)
            elif args.model==('snn-vae'or'snn-vq-vae-uni'):
                e, recon_images = model(images_spike,norm_images)
                functional.reset_net(model)
            else:
                e, recon_images,_ = model(images_spike,norm_images)
                functional.reset_net(model)
            loss_mse.append(F.mse_loss(recon_images, norm_images).item())
            ssim_loss = metric.pytorch_ssim.SSIM(window_size=11)
            loss_ssim.append((1 - ssim_loss(recon_images, norm_images)).item())
    
    print("loss_ssim = ",round(sum(loss_ssim)/len(loss_ssim),3))
    print("loss_mse = ",round(sum(loss_mse)/len(loss_mse),3))

    '''
    # 模型加法和乘法计算次数
    input = torch.randn(1, 1, 28, 28,16)
    input_ = torch.randn(1,1,28,28)

    #model = model.cpu()

    ops, params = get_model_complexity_info(model, (1, 28, 28), test_loader, as_strings=True,
                                            print_per_layer_stat=True, verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity ACs:', ops[1]))
    print('{:<30}  {:<8}'.format('Computational complexity MACs:', ops[2]))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    '''

    # 采样生成图像的质量测试
    # 生成一些样本
    if not os.path.exists("./sample/"+args.dataset_name+'/'+args.model):
        os.makedirs("./sample/"+args.dataset_name+'/'+args.model)
    sample_path = "./sample/"+args.dataset_name+'/'+args.model
    if args.model=='snn-vae':
        sampled_x, sampled_z = model.sample(batch_size_1) 
        functional.reset_net(model)  
        #print(sampled_x.shape)
        recon_images = np.array(np.clip((sampled_x + 0.5).detach().cpu().numpy(), 0., 1.) * 255, dtype=np.uint8)
        #print(recon_images.shape)
        recon_images = recon_images.reshape(4, 8, 28, 28)
        # 绘制图像
        #plt.figure(figsize=(16, 32)) # 设置画布大小
        fig = plt.figure(figsize=(10, 5), constrained_layout=True)
        gs = fig.add_gridspec(4, 8)
        for n_row in range(4):
            for n_col in range(8):
                f_ax = fig.add_subplot(gs[n_row, n_col])
                f_ax.imshow(recon_images[n_row, n_col], cmap="gray")
                f_ax.axis("off")
        
        plt.savefig(sample_path+'/image.png')
        plt.show() # 显示画布

        # 生成一堆样本用于计算IS分数
        all_images_list = []
        for i in range(40):
            sampled_x, sampled_z = model.sample(batch_size_1)
            recon_images = np.array(np.clip((sampled_x + 0.5).detach().cpu().numpy(), 0., 1.) * 255, dtype=np.uint8)
            if i==0:
                all_images=recon_images
            else:
                all_images = np.concatenate((all_images,recon_images),axis=0)
            #print(all_images.shape)
        all_images_list.append(all_images)
    
    elif args.model==('snn-vq-vae'or'snn-vq-vae-uni'):
        denoise_fn.eval() 
        temp = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        for tem in tqdm.tqdm(temp,desc='drawing', total=len(temp)):
            for k in range(20):
                sample_list=[]
                for i in range(2):
                    sample = (abdiff.sample(temp = tem, sample_steps=49)).reshape(16,7,7)
                    sample_list.append(sample)
                    functional.reset_net(denoise_fn)
                sample = torch.cat(sample_list,dim=0)
                with torch.inference_mode():
                    z = model.vq_layer.quantize(sample.cuda(0))

                    z = z.permute(0, 3, 1, 2).contiguous()

                    quantized = torch.unsqueeze(z, dim=0)
                    quantized = quantized.repeat(16, 1, 1, 1, 1)
                    quantized = model.vq_layer.poisson(quantized)
                    #torch.Size([128, 16, 7, 7, 16])

                    pred = model.decoder(quantized)
                    pred = torch.tanh(model.memout(pred))

                generated_samples = np.array(np.clip((pred + 0.5).cpu().numpy(), 0., 1.) * 255, dtype=np.uint8)
                generated_samples = generated_samples.reshape(4, 8, 28, 28)
                functional.reset_net(model)
                functional.reset_net(denoise_fn)
                fig = plt.figure(figsize=(10, 5), constrained_layout=True)
                gs = fig.add_gridspec(4, 8)
                for n_row in range(4):
                    for n_col in range(8):
                        f_ax = fig.add_subplot(gs[n_row, n_col])
                        f_ax.imshow(generated_samples[n_row, n_col], cmap="gray")
                        f_ax.axis("off")

                if not os.path.exists(sample_path+'/'+str(tem)):
                    os.makedirs(sample_path+'/'+str(tem))
                plt.savefig(sample_path+'/'+str(tem)+'/image_'+str(tem)+'_'+str(k)+'.png')
                plt.show()

        temp = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        all_images_list = []
        for tem in temp:
            for i in tqdm.tqdm(range(80), desc='Sampling_for_temp='+str(tem), total=80):
                sample = (abdiff.sample(temp = tem, sample_steps=49)).reshape(16,7,7)
                with torch.inference_mode():
                    z = model.vq_layer.quantize(sample.cuda(0))

                    z = z.permute(0, 3, 1, 2).contiguous()

                    quantized = torch.unsqueeze(z, dim=0)
                    quantized = quantized.repeat(16, 1, 1, 1, 1)
                    quantized = model.vq_layer.poisson(quantized)
                    #torch.Size([128, 16, 7, 7, 16])

                    pred = model.decoder(quantized)
                    pred = torch.tanh(model.memout(pred))

                generated_samples = np.array(np.clip((pred + 0.5).cpu().numpy(), 0., 1.) * 255, dtype=np.uint8)
                functional.reset_net(model)
                functional.reset_net(denoise_fn)
                if i==0:
                    all_images=generated_samples
                else:
                    all_images = np.concatenate((all_images,generated_samples),axis=0)
            all_images_list.append(all_images)
    
    if args.metric==None or args.metric=='IS':
        print("********now we get IS*********")
        if args.model=='snn-vae':
            print(all_images.shape)
            torch.save(all_images,'svae.pt')
            Is,_ = inception_score(np.repeat(all_images,3,axis=1)/255, cuda=True, batch_size=32, resize=True, splits=4)
            print("IS = ",Is)
        else:
            IS_list = []
            print(all_images_list[7].shape)
            torch.save(all_images_list[7],'diff.pt')
            for all_images in tqdm.tqdm(all_images_list, desc='Get IS', total=len(all_images_list)):
                Is,_ = inception_score(np.repeat(all_images,3,axis=1)/255, cuda=True, batch_size=32, resize=True, splits=4)
                IS_list.append(Is)
                #print(all_images.shape)
            print('temp = ',temp)
            print('IS = ',IS_list)

    if args.metric==None or args.metric=='KID':
        print("********now we get KID*********")
        kid = KernelInceptionDistance()
        for i, (images, labels) in enumerate(test_loader):
            if i==40:
                break
            if i==0:
                real_images = np.repeat(images,3,axis=1)
            else:
                real_image = np.repeat(images,3,axis=1)
                real_images = np.concatenate((real_images,real_image),axis=0)
        #print(real_images.shape)

        if args.model == 'snn-vae':
            A = torch.from_numpy(real_images*255).type(torch.uint8)
            B = torch.from_numpy(np.repeat(all_images,3,axis=1)).type(torch.uint8)

            kid.update(A, real=True)
            kid.update(B, real=False)
            kid_mean, kid_std = kid.compute()
            print("KID = ", kid_mean.item())
        else:
            KID_list = []
            for all_images in tqdm.tqdm(all_images_list, desc='Get KID', total=len(all_images_list)):
                A = torch.from_numpy(real_images*255).type(torch.uint8)
                B = torch.from_numpy(np.repeat(all_images,3,axis=1)).type(torch.uint8)

                kid.update(A, real=True)
                kid.update(B, real=False)
                kid_mean, kid_std = kid.compute()
                KID_list.append(kid_mean.item())
                print(kid_mean.item())
            print('temp = ',temp)
            print('KID = ',KID_list)

    if args.metric==None or args.metric=='FID':
        print("********now we get FID*********")
        Fid_list = []
        for all_images in tqdm.tqdm(all_images_list, desc='Get FID', total=len(all_images_list)):
            torch.cuda.empty_cache()
            up = nn.Upsample(size=(299, 299), mode='bilinear').type(torch.cuda.FloatTensor)
            all_images = up(torch.Tensor(all_images/255).cuda(0)).cpu().numpy()
            all_images = np.transpose(all_images,(0,2,3,1))
            all_images = np.repeat(all_images,3,axis=3)
            for i, (images, labels) in enumerate(test_loader):
                #print(i)
                if i==40:
                    break

                if i==0:
                    #real_image= tensor.new_empty((3,4), dtype=torch.int64)
                    real_image = np.repeat(images,3,axis=1)
                    real_image=up(real_image.cuda(0)).cpu().numpy()
                    real_images=np.transpose(real_image,(0,2,3,1))
                    #print(real_images.shape)
                    #print(np.max(real_images),np.min(real_images))
                else:
                    real_image = np.repeat(images,3,axis=1)
                    real_image=up(real_image.cuda(0)).cpu().numpy()
                    real_image=np.transpose(real_image,(0,2,3,1))
                    real_images = np.concatenate((real_images,real_image),axis=0)

                
            Fid = calculate_fid(all_images, real_images, use_multiprocessing=False, batch_size=4)
            print(Fid)
            Fid_list.append(Fid)
        print('Fid = ',Fid_list)







