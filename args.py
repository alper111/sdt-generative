# generator parameters
g_model = 'conv'
g_layers = [128, 64, 1]
g_depth = 4
g_proj = 'constant'
g_norm = 'batch_norm'
# discriminator parameters
d_model = 'conv'
d_layers = [64, 128, 256]
d_depth = 4
d_proj = 'constant'
d_norm = 'batch_norm'

input_shape = [256, 4, 4]
activation = 'torch.nn.ReLU()'
z_dim = 100
z_batch = 128
lr = 1e-4
lr_decay = 1.0
lr_step = 1
wasserstein = False
epoch = 50
out = "out/test/"
seed = 2019
device = 'cpu'
dataset = 'mnist'
c_iter = 1
topk = 5
acc = False
ckpt = None
