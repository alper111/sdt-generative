import argparse
import torch
import torchvision
import models, utils
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Test your models.')
parser.add_argument("-dataset", type=str)
parser.add_argument("-model_file", type=str)
parser.add_argument('-model_type', type=str)
parser.add_argument('-model_details', nargs='+', type=int, required=True)
parser.add_argument('-input_shape', nargs='+', type=int)
parser.add_argument('-normalization', type=str, default=None)
parser.add_argument('-latent_dim', type=int, required=True)
parser.add_argument('-device', type=str, required=True)
parser.add_argument('-k', type=int, default=5)
args = parser.parse_args()
device = torch.device(args.device)

if args.model_type == 'mlp':
    model = torch.nn.Sequential(models.MLP(args.model_details, torch.nn.ReLU(), normalization=args.normalization), torch.nn.Tanh())
else:
    model = torch.nn.Sequential(models.ConvDecoder(args.model_details, args.input_shape, args.latent_dim, normalization=args.normalization), torch.nn.Tanh())
state_dict = torch.load(args.model_file)
model.load_state_dict(state_dict)
model.eval()
for p in model.parameters():
    p.requires_grad = False
model.to(device)

if args.dataset == 'mnist':
    trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50)
    test_size = 10000
elif args.dataset == 'emnist':
    trainset = torchvision.datasets.EMNIST(root='./data',train=True,split='balanced',download=True,transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    trainset.train_data = trainset.train_data.permute(0, 2, 1)
    testset = torchvision.datasets.EMNIST(root='./data',train=False,split='balanced',download=True,transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    testset.test_data = testset.test_data.permute(0, 2, 1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50)
    test_size = 18800
elif args.dataset == 'fashion':
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50)
    test_size = 10000
elif args.dataset == 'cifar':
    trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transforms.ToTensor())
    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50)
    test_size = 10000
else:
    X, Y = torch.load(args.dataset)
    X = X.type(torch.float) / 255.0
    dataset_size = X.shape[0]
    dataset = torch.utils.data.TensorDataset(X)
    trainset, testset = torch.utils.data.random_split(dataset, [dataset_size-10000, 10000])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50)
    test_size = 10000

### load inception module
inception = models.InceptionV3()
for p in inception.parameters():
    p.requires_grad = False
inception.to(device)
### get real samples' inception activations
iterator = iter(testloader)
trainiterator = iter(trainloader)
x_p = trainiterator.next()[0]
real_feats = torch.empty(test_size, 2048)
fake_feats = torch.empty(test_size, 2048)
train_feats = torch.empty(test_size, 2048)
testset = torch.zeros(test_size, x_p.shape[1], x_p.shape[2], x_p.shape[3], device=device)
generatedset = torch.zeros(test_size, x_p.shape[1], x_p.shape[2], x_p.shape[3])
for i in range(test_size // 50):
    x_t = iterator.next()[0]
    real_feats[i*50:(i+1)*50] = inception(x_t.to(device)).cpu()
    generatedset[i*50:(i+1)*50] = model(torch.randn(50, args.latent_dim, device=device)).cpu().view(50, x_p.shape[1], x_p.shape[2], x_p.shape[3])*0.5+0.5
    fake_feats[i*50:(i+1)*50] = inception(generatedset[i*50:(i+1)*50].to(device)).cpu()
    train_feats[i*50:(i+1)*50] = inception(trainiterator.next()[0].to(device)).cpu()
    testset[i*50:(i+1)*50] = x_t.to(device)

fake, real = utils.nn_accuracy(p_fake=generatedset.view(test_size, -1), p_real=testset.view(test_size, -1).cpu(), k=args.k, device=torch.device('cpu'))
fid = utils.FID_score(x_fake=fake_feats, x_real=real_feats)
fid_max = utils.FID_score(x_fake=train_feats, x_real=real_feats)
print("fake: %.5f, real: %.5f, FID: %.5f, FID_orig: %.5f" % (fake, real, fid, fid_max))

