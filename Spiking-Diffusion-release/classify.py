import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms
from load_dataset_snn import *
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
#tensor = torch.rand(1280, 1, 28, 28)  # 您的张量
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(42)
class TensorDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, index):
        return self.tensor[index]

# 定义 LeNet 模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1   = nn.Linear(16*5*5, 120)
        self.fc1   = nn.Linear(256, 120) 
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 26)
        
    def forward(self, x):
        out = F.relu(self.conv1(x)) 
        out = F.max_pool2d(out, 2) 
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

svae_tensor = torch.load('svae.pt')
print(svae_tensor.shape)
test_dataset = TensorDataset(torch.tensor(svae_tensor).float()/255)
test_loader_svae = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 超参数设置
LR = 0.0001       
BATCH_SIZE = 64    
EPOCH = 10      

# MNIST 数据集
train_loader,test_loader = load_MNIST_Letters(data_path = '/data/liumingxuan/SNN-VAE-DMSVDD/datasets/Datasets',batch_size=BATCH_SIZE)

# 实例化模型
model = LeNet().cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 训练模型
for epoch in range(EPOCH):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        # 前向传播
        #print(data.shape)
        output = model(data)
        # 计算损失
        loss = criterion(output, target)
        
        # 优化器清零梯度
        optimizer.zero_grad() 
        # 损失反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch+1, loss.item()))

# 测试模型
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += criterion(output, target).item() 
        # 去掉最大的值的索引就是预测类别
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss/len(test_loader.dataset), 
    correct, len(test_loader.dataset),
    100.*correct/len(test_loader.dataset)))

pred_list=[]
with torch.no_grad():
    for data in test_loader_svae:
        data = data.cuda()
        output = model(data)
        #test_loss += criterion(output, target).item() 
        # 去掉最大的值的索引就是预测类别
        pred = output.max(1, keepdim=True)[1]
        pred_list.append(pred)

pre = torch.cat(pred_list).view(-1)
print(pre)
print(pre.shape)
hist = torch.histc(pre, bins=26, min=0, max=25) 
print(hist)

p = torch.ones(26) / 26  # 均匀分布
q = hist / pre.numel()  # 离散分布
p=p.cuda()
q=q.cuda()
kl_div = torch.sum(p * torch.log(p/q)) 
print(kl_div)

tensors = [torch.zeros(0, 1, 28, 28) for i in range(26)]

for tensor, label in zip(torch.tensor(svae_tensor), pre):
    label = int(label)  
    tensors[label] = torch.cat((tensors[label], tensor.unsqueeze(0)), 0)

for i in range(26):
    print(tensors[i].size())


for i in range(26):
    #recon_images = tensors[i][0:32].reshape(4, 8, 28, 28)
    recon_images = svae_tensor[32*i*2:32*i*2+32].reshape(4, 8, 28, 28)
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    gs = fig.add_gridspec(4, 8)
    for n_row in range(4):
        for n_col in range(8):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow(recon_images[n_row, n_col], cmap="gray")
            f_ax.axis("off")

    plt.savefig('./paper_image/image'+'_'+str(i)+'.png')
    plt.show() # 显示画布 


sizes = [tensor.size(0) for tensor in tensors]  
total_size = sum(sizes)  

result = torch.zeros(10, 1, 28, 28)  

result[0]=tensors[0][15]
result[1]=tensors[1][5]
result[2]=tensors[2][8]
result[3]=tensors[3][9]
result[4]=tensors[4][2]
result[5]=tensors[5][1]
result[6]=tensors[6][13]
result[7]=tensors[7][9]
result[8]=tensors[8][0]
result[9]=tensors[8][2]
result[0:9]=(torch.tensor(svae_tensor)[32*9*2:32*9*2+32])[16:25]
print(result.size())  # torch.Size([32, 1, 28, 28])


recon_images = result.reshape(2, 5, 28, 28)
fig = plt.figure(figsize=(10, 4), constrained_layout=True)
gs = fig.add_gridspec(2, 5)
for n_row in range(2):
    for n_col in range(5):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        #plt.tight_layout() 
        f_ax.imshow(recon_images[n_row, n_col], cmap="gray")
        f_ax.axis("off")

plt.savefig('./paper_image/image.png')
plt.show() # 显示画布 
