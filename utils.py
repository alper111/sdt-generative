import os
import time
import numpy as np
import torch
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from scipy.linalg import sqrtm

def sample_gumbel(logits):
    u = torch.rand_like(logits)
    eps = 1e-20
    u.add_(eps).log_().neg_()
    u.add_(eps).log_().neg_()
    return u 

def gumbel_softmax_sample(logits, temp=1.):
    g = sample_gumbel(logits)
    y = (g + logits) / temp
    return torch.softmax(y, dim=1)

def gumbel_softmax(logits, temp=1.):
    y = gumbel_softmax_sample(logits, temp)
    _, ind = torch.max(y, dim=1)
    y_hard = torch.eye(logits.shape[1], device=logits.device)[ind]
    y = (y_hard - y).detach() + y
    return y

def p2dist(x, y):
    y_dim = len(y.shape)
    return torch.pow(x, 2).sum(dim=-1).view(x.shape[:-1]+(1,)) - \
        2 * torch.matmul(x, y.permute(list(range(y_dim-2))+[y_dim-1, y_dim-2])) + \
            torch.pow(y, 2).sum(dim=-1).view(y.shape[:-2]+(1,y.shape[-2]))

def nn_accuracy(p_fake, p_real, device=torch.device('cpu'), k=5):
    size = p_fake.shape[0]
    p_fake = p_fake.view(size, -1).to(device)
    p_real = p_real.view(size, -1).to(device)
    p_all = torch.cat([p_fake, p_real], dim=0)
    dists = p2dist(p_all, p_all) + torch.eye(2*size, device=device) * 1e12
    values, indexes = torch.topk(dists, k=k, largest=False)

    decisions = (indexes > size-1).sum(dim=1).float() / k
    fake_acc = (size - decisions[:size].sum()).float() / size
    real_acc = (decisions[size:].sum()).float() / size
    return fake_acc.item(), real_acc.item()

def sample_pball(n,p, device=torch.device('cpu')):
    mult = torch.randn(n,p, device=device)
    direction = mult / mult.norm(dim=1).view(-1, 1)
    magnitude = (torch.pow(torch.rand(n, device=device), 1./p)).view(-1, 1)
    return direction*magnitude

def FID_score(x_real, x_fake):
    mu_real = x_real.mean(dim=0)
    mu_fake = x_fake.mean(dim=0)
    cov_real = np.cov(x_real, rowvar=False)
    cov_fake = np.cov(x_fake, rowvar=False)
    mu_diff = np.linalg.norm(mu_real-mu_fake, 2) ** 2
    covmean = 2 * sqrtm(np.matmul(cov_real,cov_fake.T))
    cov_diff = np.trace(cov_real + cov_fake - covmean)
    return mu_diff + cov_diff.real

'''gradient penalty that is used in https://arxiv.org/abs/1704.00028'''
def gradient_penalty(D, x_true, x_fake, derivative, device):
    if len(x_true.shape) == 2:
        alpha = torch.rand(x_true.size()[0],1,device=device)
    else:
        alpha = torch.rand(x_true.size()[0],1,1,1,device=device)

    alpha = alpha.expand(x_true.size())
    interpolates = alpha * x_true + (1-alpha) * x_fake
    interpolates = torch.autograd.Variable(interpolates,requires_grad=True)
    disc_interpolates = D(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size(),device=device),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2,dim=1)-derivative) ** 2).mean() * 10
    return gradient_penalty

def gradient_norm_penalty(x, func, c=1, k=1):
    x.requires_grad = True
    y = func(x)
    gradients = torch.autograd.grad(outputs=y,
                                    inputs=x,
                                    grad_outputs=torch.ones_like(y),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1)-c)**2).mean()*k
    return gradient_penalty

def one_way_gradient_penalty(D, x_true, x_fake, derivative, device):
    if len(x_true.shape) == 2:
        alpha = torch.rand(x_true.size()[0],1,device=device)
    else:
        alpha = torch.rand(x_true.size()[0],1,1,1,device=device)

    alpha = alpha.expand(x_true.size())    
    interpolates = alpha * x_true + (1-alpha) * x_fake
    interpolates = torch.autograd.Variable(interpolates,requires_grad=True)
    disc_interpolates = D(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size(),device=device),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
    gradient_penalty = (torch.relu(gradients.norm(2,dim=1)-derivative) ** 2).mean() * 10
    return gradient_penalty

# given first value, last value and epoch, computes the decay factor
def exp_decay_rate(first, last, epoch):
    return np.exp(1/epoch * np.log(last/first))

def get_parameter_count(model):
    total_num = 0
    for p in model.parameters():
        shape = p.shape
        num = 1
        for d in shape:
            num *= d
        total_num += num
    return total_num

def save_animation(name, timesteps, z, x, lims, title, alpha=1.0, s=50):
    size = x.shape[0]

    cm = np.arctan2(z[:,1], z[:,0])
    cm[cm < 0] += np.pi * 2

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].set_xlim(lims)
    ax[0].set_ylim(lims)
    ax[0].set_title("Z-dimension")
    
    ax[1].set_xlim(lims)
    ax[1].set_ylim(lims)
    ax[1].set_title(title)

    loc = abs(lims[0]-lims[1])/12
    loc = (lims[0]+loc, lims[1]-loc)
    
    def init():
        data = np.zeros((3*size,2))
        
        ax[0].set_facecolor('gray')
        ax[0].scatter(z[:,0],z[:,1],c=cm,cmap='hsv',alpha=alpha,s=s)

        ax[1].set_facecolor('gray')
        ax[1].scatter(data[:size,0], data[:size,1], c='w', alpha=alpha, s=s)
        canvas = ax[1].scatter(data[size:2*size,0], data[size:2*size,1], c=cm, cmap='hsv', alpha=alpha, s=s)

        return (canvas,)

    def animate(t):
        data = timesteps[t]
        ax[1].clear()
        ax[1].set_facecolor('gray')
        ax[1].set_xlim(lims)
        ax[1].set_ylim(lims)
        ax[1].set_title(title)
        ax[1].scatter(data[:size,0],data[:size,1],c='w', alpha=alpha, s=s)
        canvas = ax[1].scatter(data[size:2*size,0],data[size:2*size,1],c=cm,cmap='hsv',alpha=alpha,s=s)

        return (canvas,)

    anim = animation.FuncAnimation(fig, animate, init_func=init,frames=len(timesteps), interval=20, blit=True)
    anim.save(name)

def save_animation_withdisc(name, timesteps, d_field, lims, title, alpha=1.0, s=20):
    size = 2500
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(title)
    canvas = ax.scatter([], [], alpha=alpha, s=s)

    colormap = plt.get_cmap("rainbow",101)
    colorlist = [colormap(i) for i in range(101)]
    xv, yv = torch.meshgrid(torch.linspace(lims[0], lims[1], 40), torch.linspace(lims[0], lims[1], 40))
    field = torch.stack([xv.contiguous().view(-1), yv.contiguous().view(-1)], dim=1).numpy()

    colors = []
    for i in range(1600+size*2):
        if i < 1600:
            colors.append(colorlist[50])
        elif i < (1600+size):
            colors.append('tab:blue')
        else:
            colors.append('tab:orange')

    def init():
        data = np.zeros((1600+2*size, 2))
        data[:1600] = field
        data[1600:] = timesteps[0]
        canvas.set_offsets(data)
        canvas.set_color(colors)
        return (canvas,)

    def animate(t):
        data = canvas.get_offsets()
        indexes = d_field[t]
        for j in range(1600):
            colors[j] = colorlist[indexes[j]]
        data[1600:] = timesteps[t]
        canvas.set_offsets(data)
        canvas.set_color(colors)
        return (canvas,)

    anim = animation.FuncAnimation(fig, animate, init_func=init,frames=len(timesteps), interval=20, blit=True)
    anim.save(name)

