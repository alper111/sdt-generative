import argparse
import os
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import models
import utils
import time
import dataset
import matplotlib as mpl
if os.environ.get("DISPLAY","") == "":
    print("no display found. Using non-interactive Agg backend")
    mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Train a GAN.")
parser.add_argument("-g_model", help="generator model. mlp, tree or moe", type=str, required=True)
parser.add_argument("-g_layers", help="generator layer info", nargs="+", type=int, required=True)
parser.add_argument("-g_depth", help="generator tree depth", type=int)
parser.add_argument("-g_proj", help="generator leaf projection. default constant", type=str, default="constant")
parser.add_argument("-g_norm", help="generator normalization layer. batch_norm, layer_norm", type=str, default=None)
parser.add_argument("-g_drop", help="dropout rate for gatings. default 0.", type=float, default=0.0)
parser.add_argument("-d_model", help="discriminator model. mlp or conv", type=str, required=True)
parser.add_argument("-d_layers", help="discriminator layer info", nargs="+", type=int, required=True)
parser.add_argument("-d_depth", help="discriminator tree depth", type=int)
parser.add_argument("-d_proj", help="discriminator leaf projection. default constant", type=str, default="constant")
parser.add_argument("-d_norm", help="discriminator normalization layer. batch_norm, layer_norm", type=str, default=None)
parser.add_argument("-resblock", help="number of layers in each resnet block for both G and D", type=int)
parser.add_argument("-input_shape", help="if you use conv generator, this is the dimension from which gen starts deconving.", nargs="+", type=int)
parser.add_argument("-z_dim", help="dimensionality of z. default 100.", default=100, type=int)
parser.add_argument("-batch_size", help="batch size. default 128.", default=128, type=int)
parser.add_argument("-test_size", help="number of test samples. default 10000", default=10000, type=int)
parser.add_argument("-lr", help="learning rate. default 1e-4.", default=1e-4, type=float)
parser.add_argument("-lr_decay", help="decay rate of learning rate. default 1.", default=1.0, type=float)
parser.add_argument("-lr_step", help="decay step size. default 1.", default=1, type=int)
parser.add_argument("-wasserstein", help="whether to use Wasserstein GP loss or not. default 0", default=0, type=int)
parser.add_argument("-epoch", default=50, type=int)
parser.add_argument("-out", help="output folder.", type=str, required=True)
parser.add_argument("-seed", help="seed. default 2019.", default=2019, type=int)
parser.add_argument("-device", help="default cpu", default="cpu", type=str)
parser.add_argument("-dataset", default="mnist", type=str)
parser.add_argument("-c_iter", help="number of times the discriminator is trained. default 1.", type=int, default=1)
parser.add_argument("-topk", default=1, help="k-nn accuracy. default 1", type=int)
parser.add_argument("-acc", default=0, type=int)
parser.add_argument("-ckpt", help="checkpoint", type=str, default=None)
parser.add_argument("-test_step", help="test step", type=int, default=5)
parser.add_argument("-img_step", help="when to print", type=int, default=1)

args = parser.parse_args()

WASSERSTEIN = True if args.wasserstein == 1 else False
ACC = True if args.acc == 1 else False
DEVICE = torch.device(args.device)
parallel = True if torch.cuda.device_count() > 1 else False

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

### LOAD THE DATA SET ###
trainloader, testloader, train_size, test_size, num_of_classes = dataset.get_dataset(args.dataset, args.batch_size, test_batch=args.test_size)

if not os.path.exists(args.out):
    os.makedirs(args.out)
os.chdir(args.out)
arg_dict = vars(args)
for key in arg_dict.keys():
    print("%s: %s" % (key, arg_dict[key]))
    print("%s: %s" % (key, arg_dict[key]), file=open("args.txt", "a"))
print("date: %s" % time.asctime(time.localtime(time.time())))
print("date: %s" % time.asctime(time.localtime(time.time())), file=(open("args.txt", "a")))

dummy = iter(trainloader).next()[0]
num_of_channels = dummy.shape[1]
height = dummy.shape[2]
width = dummy.shape[3]
img_size = height * width
feature_size = num_of_channels * img_size
loop_per_epoch = train_size // (args.batch_size * args.c_iter)
total_loss = []
real_acc_total = []
fake_acc_total = []
fid_total = []
gen_loss_total = []
disc_loss_total = []
gen_grads_total = []
disc_grads_total = []

# generator definition
if args.g_model == "mlp":
    generator = torch.nn.Sequential(
        models.MLP(
            layer_info=args.g_layers,
            activation=torch.nn.ReLU(),
            std=None,
            normalization=args.g_norm),
        torch.nn.Tanh()
    )
elif args.g_model == "conv":
    generator = torch.nn.Sequential(
        models.ConvDecoder(
            channels=args.g_layers,
            input_shape=args.input_shape,
            latent_dim=args.z_dim,
            std=0.02,
            activation=torch.nn.ReLU(),
            normalization=args.g_norm
        ),
        torch.nn.Tanh()
    )
elif args.g_model == "resnet":
    generator = models.ResNetGenerator(
        block=models.PreActResidualBlock,
        channels=args.g_layers,
        layers=[args.resblock]*(len(args.g_layers)-1),
        input_shape=args.input_shape,
        latent_dim=args.z_dim,
        depth=args.g_depth,
        out_channels=num_of_channels,
        normalization=args.g_norm,
        parallel=parallel
    )
else:
    generator = torch.nn.Sequential(
        models.MixtureDecoder(
            channels=args.g_layers,
            input_shape=args.input_shape,
            latent_dim=args.z_dim,
            std=0.02,
            activation=torch.nn.ReLU(),
            normalization=args.g_norm,
            mixture=args.g_model,
            depth=args.g_depth,
            projection=args.g_proj,
            dropout=args.g_drop
            ),
        torch.nn.Tanh()
    )
# discriminator definition
if args.d_model == "mlp":
    discriminator = models.MLP(
        layer_info=args.d_layers,
        activation=torch.nn.ReLU(),
        std=None,
        normalization=args.d_norm)
elif args.d_model == "conv":
    discriminator = models.ConvEncoder(
        channels=args.d_layers,
        input_shape=[num_of_channels, height, width],
        latent_dim=1,
        activation=torch.nn.LeakyReLU(0.2),
        std=0.02,
        normalization=args.d_norm,
        num_classes=num_of_classes,
        parallel=parallel)
elif args.d_model == "resnet":
    discriminator = models.ResNetDiscriminator(
        block=models.PreActResidualBlock,
        channels=args.d_layers,
        layers=[args.resblock]*(len(args.d_layers)-1),
        input_shape=[num_of_channels, height, width],
        latent_dim=1,
        in_channels=num_of_channels,
        normalization=args.d_norm
    )
else:
    discriminator = models.MixtureEncoder(
        channels=args.d_layers,
        input_shape=[num_of_channels, height, width],
        latent_dim=1,
        activation=torch.nn.LeakyReLU(0.2),
        normalization=args.d_norm,
        mixture=args.d_model,
        depth=args.d_depth,
        projection=args.d_proj,
        )
        
if args.ckpt is not None:
    print("using checkpoint...")
    generator.load_state_dict(torch.load(os.path.join(args.ckpt, "gen.ckpt")))
    discriminator.load_state_dict(torch.load(os.path.join(args.ckpt, "disc.ckpt")))

generator = generator.to(DEVICE)
discriminator = discriminator.to(DEVICE)

optimG = torch.optim.Adam(lr=args.lr,params=generator.parameters(), betas=(0.5, 0.999), amsgrad=True)
optimD = torch.optim.Adam(lr=args.lr,params=discriminator.parameters(), betas=(0.5, 0.999), amsgrad=True)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizer=optimG, gamma=args.lr_decay, step_size=args.lr_step)
schedulerD = torch.optim.lr_scheduler.StepLR(optimizer=optimD, gamma=args.lr_decay, step_size=args.lr_step)
criterion = torch.nn.BCEWithLogitsLoss()
print("GENERATOR")
print(generator)
print("DISCRIMINATOR")
print(discriminator)
print("G num of params: %d" % utils.get_parameter_count(generator))
print("D num of params: %d" % utils.get_parameter_count(discriminator))

gen_avg_gradients = []
disc_avg_gradients = []
gen_dim_normalizer = []
disc_dim_normalizer = []
for p in generator.parameters():
    gen_avg_gradients.append(0.0)
    n = 1
    for d in p.shape:
        n *= d
    gen_dim_normalizer.append(n**0.5)
for p in discriminator.parameters():
    disc_avg_gradients.append(0.0)
    n = 1
    for d in p.shape:
        n *= d
    disc_dim_normalizer.append(n**0.5)

if ACC:
    ### load inception module
    inception = models.InceptionV3()
    for p in inception.parameters():
        p.requires_grad = False
    inception.to(DEVICE)
    ### get real samples' inception activations
    real_feats = torch.empty(test_size, 2048)
    iterator = iter(testloader)
    for i in range(test_size // 100):
        x_t = iterator.next()[0]*0.5+0.5
        real_feats[i*100:(i+1)*100] = inception(x_t.to(DEVICE)).cpu()
real_samples = torch.empty(test_size, feature_size)
iterator = iter(testloader)
for i in range(test_size // 100):
    x_t = iterator.next()[0]
    real_samples[i*100:(i+1)*100] = x_t.view(-1, feature_size)


print("Training starts...")
##########################
# epoch loop
for e in range(args.epoch):
    gen_avg_loss = 0.0
    disc_avg_loss = 0.0
    # batch loop
    start = time.time()
    iterator = iter(trainloader)
    for i in tqdm(range(loop_per_epoch)):
        for c in range(args.c_iter):
            # train discriminator with real data
            optimD.zero_grad()
            x_real, _ = iterator.next()
            x_real = x_real.to(DEVICE)

            if args.d_model == "mlp":
                x_real = x_real.view(-1,feature_size)
            d_real = discriminator(x_real)
            if WASSERSTEIN:
                d_real_loss = -d_real.mean()
            else:
                d_real_loss = criterion(d_real,torch.ones_like(d_real,device=DEVICE))
            # train discriminator with fake data
            x_fake = generator(torch.randn(args.batch_size, args.z_dim, device=DEVICE)).view(-1,num_of_channels,height,width)
            if args.d_model == "mlp":
                x_fake = x_fake.view(-1,feature_size)
            d_fake = discriminator(x_fake)
            if WASSERSTEIN:
                d_fake_loss = d_fake.mean()
            else:
                d_fake_loss = criterion(d_fake,torch.zeros_like(d_fake,device=DEVICE))
            
            d_loss = d_real_loss + d_fake_loss
            if WASSERSTEIN:
                d_loss += utils.gradient_penalty(discriminator, x_real, x_fake, 1.0, DEVICE)
            d_loss.backward()
            optimD.step()
            disc_avg_loss += d_loss.item()
            # log gradients
            for p_i, p in enumerate(discriminator.parameters()):
                disc_avg_gradients[p_i] += p.grad.norm().item()

        # train generator
        for p in discriminator.parameters():
            p.requires_grad = False
        optimG.zero_grad()
        x_fake = generator(torch.randn(args.batch_size, args.z_dim, device=DEVICE)).view(-1, num_of_channels, height, width)
        if args.d_model == "mlp":
            x_fake = x_fake.view(-1, feature_size)
        g_loss = discriminator(x_fake)
        if WASSERSTEIN:
            g_loss = -g_loss.mean()
        else:
            g_loss = criterion(g_loss,torch.ones_like(g_loss,device=DEVICE))
        g_loss.backward()
        optimG.step()
        gen_avg_loss += g_loss.item()
        # log gradients
        for p_i, p in enumerate(generator.parameters()):
            gen_avg_gradients[p_i] += p.grad.norm().item()
        for p in discriminator.parameters():
            p.requires_grad = True
    finish = time.time()
    schedulerG.step()
    schedulerD.step()
    gen_loss_total.append(gen_avg_loss/loop_per_epoch)
    disc_loss_total.append(disc_avg_loss/(loop_per_epoch*args.c_iter))
    print("epoch: %d - disc loss: %.5f - gen loss: %.5f - time elapsed: %.3f" % (e+1, disc_loss_total[-1], gen_loss_total[-1], finish-start))
    
    ### accumulate gradients
    for g_i in range(len(gen_avg_gradients)):
        gen_avg_gradients[g_i] = gen_avg_gradients[g_i] / (loop_per_epoch * gen_dim_normalizer[g_i])
    for g_i in range(len(disc_avg_gradients)):
        disc_avg_gradients[g_i] = disc_avg_gradients[g_i] / (loop_per_epoch*args.c_iter*disc_dim_normalizer[g_i])
    # print("g avg grads:", gen_avg_gradients)
    gen_grads_total.append(gen_avg_gradients.copy())
    disc_grads_total.append(disc_avg_gradients.copy())
    for g_i in range(len(gen_avg_gradients)):
        gen_avg_gradients[g_i] = 0
    for g_i in range(len(disc_avg_gradients)):
        disc_avg_gradients[g_i] = 0
    ###

    if e+1 == 1:
        epoch_time = finish - start
        eta = epoch_time * args.epoch
        finish = time.asctime(time.localtime(time.time()+eta))
        print("### set your alarm at:",finish,"###")

    if (e+1) % args.img_step == 0:
        with torch.no_grad():
            generator.eval()
            samples = generator(torch.randn(100, args.z_dim, device=DEVICE)).cpu().detach() * 0.5 + 0.5
            if args.g_model == "mlp":
                samples = samples.view(-1, num_of_channels, height, width)
            torchvision.utils.save_image(samples, "gan_{0}.png".format(e+1), nrow=10)
            generator.train()

    # 1-nn accuracy
    if (e+1) % args.test_step == 0:
        with torch.no_grad():
            generator.eval()
            discriminator.eval()
            print("calculating nn accuracy...")
            fake_samples = torch.empty(test_size, num_of_channels, height, width)
            if ACC:
                fake_feats = torch.empty(test_size, 2048)
            for xx in range(test_size // 100):
                fake_samples[xx*100:(xx+1)*100] = generator(torch.randn(100, args.z_dim, device=DEVICE)).cpu().detach().view(-1, num_of_channels, height, width)*0.5+0.5
                if ACC:
                    fake_feats[xx*100:(xx+1)*100] = inception(fake_samples[xx*100:(xx+1)*100].to(DEVICE)).cpu()
            if ACC:
                fid = utils.FID_score(x_real=real_feats, x_fake=fake_feats)
                fake_acc, real_acc = utils.nn_accuracy(p_fake=fake_feats.to(DEVICE), p_real=real_feats.to(DEVICE), device=DEVICE, k=args.topk)
            else:
                fid = -1
                fake_samples = fake_samples.view(-1,feature_size)
                fake_acc, real_acc = utils.nn_accuracy(p_fake=fake_samples.to(DEVICE), p_real=real_samples.to(DEVICE)*0.5+0.5, device=DEVICE, k=args.topk)
        
        print("fake acc: %.5f - real acc: %.5f - FID: %.5f" % (fake_acc, real_acc, fid))
        fake_acc_total.append(fake_acc)
        real_acc_total.append(real_acc)
        fid_total.append(fid)

        # saving statistics
        np.save("fa.npy", fake_acc_total)
        np.save("ra.npy", real_acc_total)
        np.save("genloss.npy", gen_loss_total)
        np.save("discloss.npy", disc_loss_total)
        np.save("fre.npy", fid_total)
        np.save("disc_grads.npy", disc_grads_total)
        np.save("gen_grads.npy", gen_grads_total)
        torch.save(generator.cpu().state_dict(), "gen.ckpt")
        torch.save(discriminator.cpu().state_dict(), "disc.ckpt")
        generator.to(DEVICE)
        discriminator.to(DEVICE)
        generator.train()
        discriminator.train()

generator.eval()
discriminator.eval()
torch.save(generator.cpu().state_dict(),"gen.ckpt")
torch.save(discriminator.cpu().state_dict(),"disc.ckpt")

plt.plot(fake_acc_total)
plt.plot(real_acc_total)
plt.plot((np.array(fake_acc_total)+np.array(real_acc_total))*0.5,"--")
plt.legend(["fake acc.", "real acc.","total acc."])
pp = PdfPages("accuracy.pdf")
pp.savefig()
pp.close()
plt.close()

plt.plot(disc_loss_total)
plt.plot(gen_loss_total)
plt.legend(["disc. loss", "gen. loss"])
pp = PdfPages("loss.pdf")
pp.savefig()
pp.close()
plt.close()

plt.plot(fid_total)
pp = PdfPages("fid.pdf")
pp.savefig()
pp.close()
plt.close()

