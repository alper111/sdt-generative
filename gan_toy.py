import argparse
import os, time
import numpy as np
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import models
import utils
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train a generator on artificial data set.')
parser.add_argument('-lr', default=1e-3, type=float)
parser.add_argument('-batch_size',help='batch size. default 16',default=16,type=int)
parser.add_argument('-epoch',default=300,type=int)
parser.add_argument('-out',help='output folder.',type=str,required=True)
parser.add_argument('-title',help='title of the animation.',type=str,required=True)
parser.add_argument('-seed',help='seed. default 2019.',default=2019,type=int)
parser.add_argument('-dataset', help='grid, spiral, ellipse, unbalanced.', default='grid', type=str)
parser.add_argument('-wasserstein', default=0, type=int)
parser.add_argument('-c_iter', default=1, type=int)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

WASSERSTEIN = True if args.wasserstein==1 else False
NUM_OF_POINTS = 2500
BATCH_SIZE = args.batch_size
NUM_OF_EPOCHS = args.epoch
out_directory = args.out
print(args)

if not os.path.exists(out_directory):
    os.makedirs(out_directory)

print(args,file=open(out_directory+"args.txt","w"))

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.dataset == 'grid':
    x_axis = np.linspace(-10, 10, 5)
    y_axis = np.linspace(-10, 10, 5)
    it = 0
    x = torch.empty(NUM_OF_POINTS, 2, dtype=torch.float)
    CLUSTER_SIZE = NUM_OF_POINTS // 25
    for i in range(5):
       for j in range(5):
           x[it*CLUSTER_SIZE:(it+1)*CLUSTER_SIZE,0] = torch.randn(CLUSTER_SIZE) * 0.05 + x_axis[i]
           x[it*CLUSTER_SIZE:(it+1)*CLUSTER_SIZE,1] = torch.randn(CLUSTER_SIZE) * 0.05 + y_axis[j]
           it += 1
elif args.dataset == 'ellipse':
    r = 4
    th = torch.rand(NUM_OF_POINTS) * np.pi * 2.0
    x = torch.empty(NUM_OF_POINTS, 2, dtype=torch.float)
    x[:,0] = r * torch.cos(th)
    x[:,1] = r * torch.sin(th)
    x = torch.matmul(x, torch.randn(2, 2) * 0.5)+torch.randn(2)
elif args.dataset == 'spiral':
    r = torch.sqrt(torch.linspace(0, 1, NUM_OF_POINTS)) * 780 * (2*np.pi)/360
    dx = -torch.cos(r)*r + torch.rand(NUM_OF_POINTS) * (0.5)
    dy = torch.sin(r)*r + torch.rand(NUM_OF_POINTS) * (0.5)
    x = torch.stack([dx, dy]).t()
elif args.dataset == 'unbalanced':
    x = torch.empty(NUM_OF_POINTS, 2, device=device)
    x[:1250] = torch.randn(NUM_OF_POINTS//2, 2, device=device) * 0.25 + torch.tensor([-5., 5.], device=device)
    x[1250:] = torch.randn(NUM_OF_POINTS//2, 2, device=device) * 2 + torch.tensor([5., -5.], device=device)
x = x.to(device)

z = torch.randn(NUM_OF_POINTS, 2, device=device)

# generator = models.SoftTree(in_features=2, out_features=2, depth=5, projection='linear')
# discriminator = models.SoftTree(in_features=2, out_features=1, depth=5, projection='linear')
generator = models.MLP(layer_info=[2, 400, 400, 400, 400, 2], activation=torch.nn.ReLU(), normalization=None)
discriminator = models.MLP(layer_info=[2, 400, 400, 400, 400, 1], activation=torch.nn.ReLU(), normalization="layer_norm")

generator.to(device)
discriminator.to(device)

print("GENERATOR")
print(generator)
print("DISCRIMINATOR")
print(discriminator)
print("G num of params: %d" % utils.get_parameter_count(generator))
print("D num of params: %d" % utils.get_parameter_count(discriminator))

optimG = torch.optim.Adam(lr=args.lr, betas=(0.5, 0.999), params=generator.parameters(), amsgrad=True)
optimD = torch.optim.Adam(lr=args.lr, betas=(0.5, 0.999), params=discriminator.parameters(), amsgrad=True)
criterion = torch.nn.BCEWithLogitsLoss()

print("Training starts...")

size = x.shape[0]
critic_iter = args.c_iter
loop_per_epoch = size // (BATCH_SIZE * critic_iter)
total_loss = torch.zeros(NUM_OF_EPOCHS)
timesteps = []
real_total = []
fake_total = []
fid_total = []
disc_total = []
gen_total = []

for e in range(NUM_OF_EPOCHS):

    R = torch.randperm(size)
    gen_avg_loss = 0.0
    disc_avg_loss = 0.0
    g_count = 0
    d_count = 0
    
    start_time = time.time()
    for i in tqdm(range(loop_per_epoch)):
        for c in range(critic_iter):
            # train discriminator with real data
            optimD.zero_grad()
            x_real = x[R[(critic_iter*i+c)*args.batch_size:(critic_iter*i+c+1)*args.batch_size]]
            x_real = x_real.to(device)
            d_real = discriminator(x_real)
            # wasserstein
            if WASSERSTEIN:
                d_real_loss = -d_real.mean()
            # original
            else:
                d_real_loss = criterion(d_real, torch.ones_like(d_real,device=device))
            
            # train discriminator with fake data
            x_fake = generator(torch.randn(args.batch_size, 2, device=device))
            d_fake = discriminator(x_fake)
            # wasserstein
            if WASSERSTEIN:
                d_fake_loss = d_fake.mean()
            # original
            else:
                d_fake_loss = criterion(d_fake, torch.zeros_like(d_fake,device=device))

            d_loss = d_real_loss + d_fake_loss
            # wasserstein
            if WASSERSTEIN:
                d_loss += utils.gradient_penalty(discriminator,x_real,x_fake,1.0,device)
            d_loss.backward()
            optimD.step()
            disc_avg_loss += d_loss.item()
            d_count += 1

        # train generator
        for p in discriminator.parameters():
            p.requires_grad = False
        optimG.zero_grad()

        x_fake = generator(torch.randn(args.batch_size, 2, device=device))
        g_loss = discriminator(x_fake)
        # wasserstein
        if WASSERSTEIN:
            g_loss = -g_loss.mean()
        else:
            g_loss = criterion(g_loss,torch.ones_like(g_loss, device=device))
        g_loss.backward()
        optimG.step()
        gen_avg_loss += g_loss.item()
        g_count += 1

        for p in discriminator.parameters():
            p.requires_grad = True
    finish_time = time.time()

    print("epoch: %d - disc loss: %.5f - gen loss: %.5f - time elapsed: %.5f" % (e+1, disc_avg_loss / d_count, gen_avg_loss / g_count, finish_time-start_time))
    gen_total.append(gen_avg_loss/g_count)
    disc_total.append(disc_avg_loss/d_count)
    
    if (e+1) % args.c_iter == 0:
        generator.eval()
        with torch.no_grad():
            data = np.zeros((size * 2,2))
            data[:size] = x.cpu()
            data[size:] = generator(z).cpu().numpy()
            timesteps.append(data)

            fake_acc, real_acc = utils.nn_accuracy(p_real=x, p_fake=generator(z), k=5)
            fid = utils.FID_score(x.cpu(), generator(z).cpu())
            print("fake acc: %.5f - real acc: %.5f - FID: %.5f" % (fake_acc, real_acc, fid))
            fake_total.append(fake_acc)
            real_total.append(real_acc)
            fid_total.append(fid)
        generator.train()

plt.plot(fake_total)
plt.plot(real_total)
plt.plot((np.array(fake_total)+np.array(real_total))*0.5, '--')
plt.legend(["fake acc.", "real acc.", "total acc."])
pp = PdfPages(out_directory+'accuracy.pdf')
pp.savefig()
pp.close()
plt.close()

plt.plot(disc_total)
plt.plot(gen_total)
plt.legend(["disc. loss", "gen. loss"])
pp = PdfPages(out_directory+'loss.pdf')
pp.savefig()
pp.close()
plt.close()

plt.plot(fid_total)
pp = PdfPages(out_directory+'fid.pdf')
pp.savefig()
pp.close()
plt.close()

torch.save(generator.cpu().state_dict(),out_directory+'gen.ckpt')
torch.save(discriminator.cpu().state_dict(),out_directory+'disc.ckpt')
np.save(out_directory+"fid.npy", fid_total)
np.save(out_directory+"fake.npy", fake_total)
np.save(out_directory+"real.npy", real_total)
np.save(out_directory+"g_loss.npy", gen_total)
np.save(out_directory+"d_loss.npy", disc_total)

utils.save_animation(name=out_directory+'animation.mp4',timesteps=timesteps, z=z.cpu(), x=x.cpu(), lims=(-15, 15), title=args.title, alpha=0.1)
