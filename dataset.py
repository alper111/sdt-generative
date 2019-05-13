import torch
import torchvision
import torchvision.transforms as transforms

class BinaryDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, return_idx=False):
        x, y = torch.load(root)
        self.data = x
        self.labels = y
        self.root = root
        self.transform = transform
        self.return_idx = return_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_t = self.data[idx].type(torch.float)
        y_t = self.labels[idx]
        if self.transform:
            x_t = self.transform(x_t)
        if self.return_idx:
            return (x_t, y_t, idx)
        else:
            return (x_t, y_t)

class IndexedDataset(torch.utils.data.Dataset):
    """ 
    Wraps another dataset to sample from. Returns the sampled indices during iteration.
    In other words, instead of producing (X, y) it produces (X, y, idx)
    source: https://github.com/tneumann/minimal_glo/blob/master/glo.py
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return (img, label, idx)

def get_dataset(name, batch_size, embedding=False, return_idx=False):

    if name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
        trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
        testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)
        if return_idx:
            trainset = IndexedDataset(trainset)
            testset = IndexedDataset(testset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=50, num_workers=4)
        train_size = 60000
        test_size = 10000
        num_of_classes = 10
    elif name == 'emnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
        trainset = torchvision.datasets.EMNIST(root='./data',train=True,split='balanced',download=True,transform=transform)
        testset = torchvision.datasets.EMNIST(root='./data',train=False,split='balanced',download=True,transform=transform)
        trainset.train_data = trainset.train_data.permute(0, 2, 1)
        testset.test_data = testset.test_data.permute(0, 2, 1)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=50, num_workers=4)
        train_size = 112800
        test_size = 10000
        num_of_classes = trainset.targets.max().item()+1
    elif name == 'fashion':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=50, num_workers=4)
        train_size = 60000
        test_size = 10000
        num_of_classes = trainset.targets.max().item()+1
    elif name == 'cifar':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=50, num_workers=4)
        train_size = 50000
        test_size = 10000
        num_of_classes = trainset.targets.max().item()+1
    elif name == 'stl':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        trainset = torchvision.datasets.STL10(root='./data', split='train', download=True,transform=transform)
        testset = torchvision.datasets.STL10(root='./data', split='test', download=True,transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=50, num_workers=4)
        train_size = 5000
        test_size = 8000
        num_of_classes = trainset.targets.max().item()+1
    else:
        if embedding:
            X = torch.load(name)
            dev = torch.device('cuda:0')
            X = X.to(dev)
            mu = X.mean(dim=0)
            std = X.std(dim=0)
            X = ((X-mu)/std).cpu()
            dataset_size = X.shape[0]
            Y = torch.zeros(dataset_size, dtype=torch.int)
            dataset = torch.utils.data.TensorDataset(X, Y)
        else:
            dataset = BinaryDataset(name, transform=transforms.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]), return_idx=return_idx)
            dataset_size = dataset.__len__()
        R = torch.randperm(dataset_size)
        train_indices = torch.utils.data.SubsetRandomSampler(R[10000:])
        test_indices = torch.utils.data.SubsetRandomSampler(R[:10000])
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_indices)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=50, sampler=test_indices)
        num_of_classes = 1
        train_size = dataset_size-10000
        test_size = 10000
    
    if embedding:
        return trainloader, testloader, train_size, test_size, num_of_classes, mu, std
    else:
        return trainloader, testloader, train_size, test_size, num_of_classes
