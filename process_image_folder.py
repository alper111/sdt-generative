import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str,	required=True)
parser.add_argument('--output_folder', type=str, required=True)
parser.add_argument('--output_name', type=str, required=True)
parser.add_argument('--crop_size', type=int, required=True)
parser.add_argument('--image_size', type=int, required=True)
args = parser.parse_args()

transform = transforms.Compose([transforms.CenterCrop(args.crop_size), transforms.Resize(args.image_size),transforms.ToTensor()])
dataset = torchvision.datasets.ImageFolder(root=args.image_folder,transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=100, num_workers=4)
N = len(dataset.samples)

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

data = torch.zeros(N, 3, args.image_size, args.image_size, dtype=torch.uint8)
labels = torch.zeros(N, dtype=torch.int32)
for i, (x, y) in enumerate(loader,0):
    print("batch %d" % i)
    if x.shape[0] != 100:
    	data[i*100:] = (x * 255).type(torch.uint8)
    	labels[i*100:] = y.type(torch.int32)
    else:
    	data[i*100:(i+1)*100] = (x * 255).type(torch.uint8)
    	labels[i*100:(i+1)*100] = y.type(torch.int32)
torch.save((data, labels), os.path.join(args.output_folder, args.output_name))

