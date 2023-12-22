import os
import torchvision.datasets
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
import torch
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
import random
import numpy as np

def load_mnist(data_path,batch_size):
    #print("loading MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    batch_size = batch_size
    input_size = 28
    
    #SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    

    transform_train = transforms.Compose([
        #transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        #SetRange
    ])
    
    transform_test = transforms.Compose([
        #transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        #SetRange
    ])
    
    # 加载train和test的set
    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform_test, download=True)
    
    # 加载trainloader与testloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def load_fashionmnist(data_path,batch_size):
    #print("loading Fashion MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = batch_size
    input_size = 28

    #SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        #transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        #SetRange
    ])

    transform_test = transforms.Compose([
        #transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        #SetRange
    ])

    trainset = torchvision.datasets.FashionMNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.FashionMNIST(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    return trainloader, testloader

def load_cifar10(data_path,batch_size):
    #print("loading CIFAR10")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = batch_size
    input_size = 28

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.CIFAR10(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.CIFAR10(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
    return trainloader, testloader

def load_celebA(data_path,batch_size):
    #print("loading CelebA")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = batch_size
    input_size = 28
    
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange])

    trainset = torchvision.datasets.CelebA(root=data_path, 
                                            split='train', 
                                            download=True, 
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers=8, pin_memory=True)

    testset = torchvision.datasets.CelebA(root=data_path, 
                                            split='test', 
                                            download=True, 
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size, 
                                            shuffle=False, num_workers=8, pin_memory=True)
    return trainloader, testloader
 
def load_KMNIST(data_path,batch_size):
    #print("loading KMNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    batch_size = batch_size
    input_size = 28
    

    #SetRange = transforms.Lambda(lambda X: 2 * X - 1.)

    transform_train = transforms.Compose([
        #transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        #SetRange
    ])
    

    transform_test = transforms.Compose([
        #transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        #SetRange
    ])
    
    # 加载train和test的set
    trainset = torchvision.datasets.KMNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.KMNIST(data_path, train=False, transform=transform_test, download=True)
    
    # 加载trainloader与testloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

# 加载MNIST-square数据集
def square_creation(input_tensor: torch.Tensor):
    posible_values = [2, 20]
    mean = int(torch.mean(input_tensor[0])*100)
    random.seed(mean)
    x_rnd = random.randint(0, 1)
    x_start = posible_values[x_rnd]
    random.seed(mean-1)
    y_rnd = random.randint(0, 1)
    y_start = posible_values[y_rnd]
    input_tensor[:, x_start:x_start+6, y_start:y_start +
                 6] = torch.ones((1, 6, 6), dtype=torch.float32)
    return input_tensor

def load_MNIST_square(data_path,batch_size):
    
    # 从config中拿到batch_size和input_size两个数据
    batch_size = batch_size
    input_size = 28
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = torchvision.transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            SetRange,
            transforms.Normalize((0,), (1,)),
            torchvision.transforms.Lambda(square_creation)
        ]
    )
    
    test_data_MNIST_square = torchvision.datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader_MNIST_square = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=data_path,
            train=False,
            transform=transform,
        ),
        batch_size=batch_size
    )
    return test_loader_MNIST_square

# 加载CIFAR10_BW数据集
def load_CIFAR10_BW(data_path,batch_size):
    
    # 从config中拿到batch_size和input_size两个数据
    batch_size = batch_size
    input_size = 28
    
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((input_size, input_size)),
            SetRange
        ]
    )

    test_data_CIFAR10 = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=False,
        transform=transform,
    )

    test_loader_CIFAR10 = torch.utils.data.DataLoader(
        test_data_CIFAR10,
        batch_size=batch_size,
        shuffle=False
    )
    return test_loader_CIFAR10

# 加载Letters数据集
def load_MNIST_Letters(data_path,batch_size):
    
    # 从config中拿到batch_size和input_size两个数据
    batch_size = batch_size
    input_size = 28
    #SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = torchvision.transforms.Compose(
        [
            #transforms.Resize((input_size, input_size)),
            lambda img: torchvision.transforms.functional.rotate(img, -90),
            lambda img: torchvision.transforms.functional.hflip(img),
            torchvision.transforms.ToTensor(),
            #SetRange
        ]
    )
    test_data_letters = torchvision.datasets.EMNIST(
        root=data_path,
        split="letters",
        train=False,
        download=False,
        transform=transform,
        target_transform=Lambda(
            lambda y: y-1
        ),
    )
    # Eliminate the first class, that is non-existant for our case
    test_data_letters.classes = test_data_letters.classes[1:]
    test_loader_letters = torch.utils.data.DataLoader(
        test_data_letters,
        batch_size=batch_size,
        shuffle=False
    )
    # To obtain 10.000 test samples, pass trought the function
    #test_loader_letters = parse_size_of_dataloader(test_loader_letters, batch_size)
    if 1:
        train_data_letters = torchvision.datasets.EMNIST(
            root=data_path,
            split="letters",
            train=True,
            download=False,
            transform=transform,
            target_transform=Lambda(
                lambda y: y-1
            ),
        )
        train_data_letters.classes = train_data_letters.classes[1:]
        train_loader_letters = torch.utils.data.DataLoader(
            train_data_letters,
            batch_size=batch_size,
            shuffle=True
        )

        return train_loader_letters, test_loader_letters
    
# 加载notMNIST数据集
class notMNIST(VisionDataset):
    def __init__(self, root: Path, transform=None, samples_per_class=None):
        """
        Args:
            root (string): Directory with the images of the selected option.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        classes = []
        images = []
        targets = []
        # Every directory is a class in the structure of notMNINST
        for cl_index, class_dir_name in enumerate(sorted(root.iterdir())):
            # Extract the class name from the path
            classes.append(class_dir_name.as_posix().split('/')[-1])
            class_dir_path = root / class_dir_name
            if samples_per_class is None:
                # We put the limit way above the length of the dataset to not limit
                # the data collected at all, as None indicates that we want all the
                # data available
                limit = 1000000
            else:
                limit = samples_per_class
            for index, png_im in enumerate(sorted(class_dir_path.iterdir())):
                if index == limit:
                    break
                else:
                    # Some images are corrupted, so we skip those images
                    try:
                        # Get images in range 0-1
                        images.append(torchvision.io.read_image(
                            png_im.as_posix())/255)
                        targets.append(cl_index)
                    except RuntimeError:
                        # we have to update the limit to obtain the desired
                        # number of imgs per class
                        limit += 1
                        continue

        # Transform list to tensors
        images = torch.stack(images, dim=0)
        targets = torch.tensor(targets, dtype=torch.uint8)
        
        # Shuffle fixed for reproducibility reasons
        self.images = images
        self.targets = targets
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        trans = transforms.Compose([
              transforms.Resize((32, 32)),
              SetRange
        ])
        
        self.images = trans(self.images)
            
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [self.images[idx], self.targets[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample

def load_notMNIST(data_path, samples_per_class=1000):
    # Test only for compatibility issues
    
    root_path = Path(data_path)
    notmnist_path = root_path  / 'notMNIST_small'
    batch_size = 128
    input_size = 28
    loader = torch.utils.data.DataLoader(
        notMNIST(notmnist_path, samples_per_class=samples_per_class),
        batch_size=batch_size,
        shuffle=False
    )
    return loader    

class MNIST_C(VisionDataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with the images of the selected option.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = Path(root_dir)
        self.images = np.load(self.root / 'test_images.npy')
        self.targets = np.load(self.root / 'test_labels.npy').astype('uint8')
        
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            SetRange,
            transforms.Normalize((0,), (1,))])
            
        self.transform = transform
        self.classes = [str(x) for x in range(10)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [self.images[idx], self.targets[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))/255
        return [torch.from_numpy(image), torch.tensor(label)]
    
def load_MNIST_C(data_path, option='zigzag'):
    #if not (datasets_path / 'mnist_c').exists():
        #!wget - O mnist_c.gz - -no-check-certificate "https://tecnalia365-my.sharepoint.com/:u:/g/personal/aitor_martinez_tecnalia_com/ERi3c4DxluJFqpv4wtlTkKEBvhdrY4WwqNRJWKyyVoTQqg?download=1"
        #!unzip mnist_c.gz - d $datasets_path
        #!rm mnist_c.gz
        #print('please download MNIST-C from "https://tecnalia365-my.sharepoint.com/:u:/g/personal/aitor_martinez_tecnalia_com/ERi3c4DxluJFqpv4wtlTkKEBvhdrY4WwqNRJWKyyVoTQqg?download=1"!')

    mnist_c_loader = torch.utils.data.DataLoader(
        MNIST_C(datasets_path / 'mnist_c' / option, ToTensor()),
        batch_size=256,
        shuffle=False
    )
    return mnist_c_loader
