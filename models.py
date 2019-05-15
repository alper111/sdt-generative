import numpy
import math
import torch
import torchvision

'''linear layer with optional batch normalization or layer normalization'''
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, std=None, normalization=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        self.normalization = normalization
        if normalization == 'batch_norm':
            self.normalization_func = torch.nn.BatchNorm1d(num_features=self.out_features)
        elif normalization == 'layer_norm':
            self.normalization_func = torch.nn.LayerNorm(normalized_shape=self.out_features)

        if std is not None:
            self.weight.data.normal_(0., std)
            self.bias.data.normal_(0., std)
        else:
            # he initialization for ReLU activaiton
            stdv = math.sqrt(2 / self.weight.size(1))
            self.weight.data.normal_(0., stdv)
            self.bias.data.zero_()

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.normalization:
            x = self.normalization_func(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, normalization={}'.format(
            self.in_features, self.out_features, self.normalization
        )
        
'''convolutional layer with std option'''
class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, std=None, bias=True):
        super(Conv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if std is not None:
            self.conv.weight.data.normal_(0., std)
            self.conv.bias.data.normal_(0., std)

    def forward(self, x):
        x = self.conv(x)
        return x

'''convolution transpose layer with optional batch normalization'''
class ConvTranspose2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, std=None, bias=True):
        super(ConvTranspose2d, self).__init__()
        self.convt = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if std is not None:
            self.convt.weight.data.normal_(0., std)
            self.convt.bias.data.normal_(0., std)

    def forward(self, x):
        x = self.convt(x)
        return x

'''DCGAN-like convolutional encoder'''
class ConvEncoder(torch.nn.Module):
    def __init__(self, channels, input_shape, latent_dim, activation=torch.nn.ReLU(), std=None, normalization=None, conditional=False, num_classes=10):
        super(ConvEncoder, self).__init__()
        self.conditional = conditional
        convolutions = []
        current_shape = input_shape
        for ch in channels:
            convolutions.append(Conv2d(in_channels=current_shape[0], out_channels=ch, kernel_size=4, stride=2, padding=1, std=std))
            current_shape = [ch, current_shape[1] // 2, current_shape[2] // 2]
            if normalization == 'batch_norm':
                convolutions.append(torch.nn.BatchNorm2d(ch))
            elif normalization == 'layer_norm':
                convolutions.append(torch.nn.LayerNorm(current_shape))
            convolutions.append(activation)
        self.convolutions = torch.nn.Sequential(*convolutions)
        self.dense = Linear(in_features=current_shape[0] * current_shape[1] * current_shape[2], out_features=latent_dim, std=std, normalization=None)
        if conditional:
            self.dense_cond = torch.nn.Linear(in_features=current_shape[0] * current_shape[1] * current_shape[2], out_features=num_classes, bias=False)

    def forward(self, x, y=None):
        out = self.convolutions(x)
        out = out.view(out.shape[0], -1)
        if self.conditional:
            y_bar = self.dense_cond(out)
            similarity = (y * y_bar).sum(dim=1)
            out = self.dense(out) + similarity.view(-1,1)
        else:
            out = self.dense(out)
        return out

'''DCGAN-like convolutional decoder'''
class ConvDecoder(torch.nn.Module):
    def __init__(self, channels, input_shape, latent_dim, activation=torch.nn.ReLU(), std=None, normalization=None):
        super(ConvDecoder, self).__init__()
        self.input_shape = input_shape
        self.dense = torch.nn.Sequential(
            Linear(in_features=latent_dim, out_features=input_shape[0] * input_shape[1] * input_shape[2], std=std, normalization=normalization),
            activation)

        convolutions = []
        current_shape = input_shape
        for ch in channels[:-1]:
            convolutions.append(ConvTranspose2d(in_channels=current_shape[0], out_channels=ch, kernel_size=4, stride=2, padding=1, std=std))
            current_shape = [ch, current_shape[1] * 2, current_shape[2] * 2]
            if normalization == 'batch_norm':
                convolutions.append(torch.nn.BatchNorm2d(ch))
            elif normalization == 'layer_norm':
                convolutions.append(torch.nn.LayerNorm(current_shape))
            convolutions.append(activation)
        convolutions.append(ConvTranspose2d(in_channels=current_shape[0], out_channels=channels[-1], kernel_size=4, stride=2, padding=1, std=std))
        self.convolutions = torch.nn.Sequential(*convolutions)

    def forward(self, x):
        out = self.dense(x)
        out = out.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        out = self.convolutions(out)
        return out

'''multi-layer perceptron with batch norm option'''
class MLP(torch.nn.Module):
    def __init__(self, layer_info, activation, std=None, normalization=None, conditional=False, num_classes=10):
        super(MLP, self).__init__()
        self.conditional = conditional
        layers = []
        in_dim = layer_info[0]
        for l in layer_info[1:-1]:
            layers.append(Linear(in_features=in_dim, out_features=l, std=std, normalization=normalization))
            layers.append(activation)
            in_dim = l
        layers.append(Linear(in_features=in_dim, out_features=layer_info[-1], std=std, normalization=None))
        if conditional:
            self.layers = torch.nn.Sequential(*layers[:-1])
            self.dense = layers[-1]
            self.dense_cond = torch.nn.Linear(in_features=layer_info[-2], out_features=num_classes, bias=False)
        else:
            self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, y=None):
        if self.conditional:
            feats = self.layers(x)
            y_bar = self.dense_cond(feats)
            similarity = (y * y_bar).sum(dim=1)
            out = self.dense(feats) + similarity.view(-1, 1)
        else:
            out = self.layers(x)
        return out

class HighwayBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, activation=torch.nn.ReLU(), bias_init=-1.):
        super(HighwayBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # transform gate
        self.T = Linear(in_features=in_features, out_features=out_features)
        with torch.no_grad():
            self.T.bias.data.fill_(bias_init)
        # block state
        self.H = Linear(in_features=in_features, out_features=out_features)
        self.act = activation
        
    def forward(self, x):
        T = torch.sigmoid(self.T(x))
        H = self.act(self.H(x))
        return H * T + (1-T) * x

class HighwayNet(torch.nn.Module):
    def __init__(self, in_features, out_features, num_layers, activation=torch.nn.ReLU(), bias_init=-1.):
        super(HighwayNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            layers.append(HighwayBlock(in_features, out_features, activation=activation, bias_init=bias_init))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# original residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, resample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.resample = resample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.resample:
            residual = self.resample(x)
        out += residual
        out = self.relu(out)
        return out

# pre-activation residual block
class PreActResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, resample=None):
        super(PreActResidualBlock, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.resample = resample
        if in_channels > out_channels:
            self.conv1x1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)


    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        # map residual to reduce channels with 1x1 convolution
        if residual.shape[1] > out.shape[1]:
            residual = self.conv1x1(residual)
        # increase residual's channels with zeros.
        elif residual.shape[1] < out.shape[1]:
            ch_num = out.shape[1] - residual.shape[1]
            zero_channel = torch.zeros(residual.shape[0], ch_num, residual.shape[2], residual.shape[3], device=x.device)
            residual = torch.cat([residual, zero_channel], dim=1)
        out += residual
        if self.resample == 'upsample':
            out = torch.nn.functional.interpolate(out, scale_factor=2)
        elif self.resample == 'downsample':
            out = torch.nn.functional.interpolate(out, scale_factor=0.5)
        return out

# ResNet
class ResNetGenerator(torch.nn.Module):
    def __init__(self, block, channels, layers, input_shape, latent_dim):
        super(ResNetGenerator, self).__init__()

        self.input_shape = input_shape
        self.dense = torch.nn.Linear(latent_dim, input_shape[0]*input_shape[1]*input_shape[2], bias=False)
        self.layer1 = self.make_layer(block, in_channels=channels[0], out_channels=channels[1], blocks=layers[0], resample='upsample')
        self.layer2 = self.make_layer(block, in_channels=channels[1], out_channels=channels[2], blocks=layers[1], resample='upsample')
        self.layer3 = self.make_layer(block, in_channels=channels[2], out_channels=channels[3], blocks=layers[2], resample='upsample')
        self.layer4 = self.make_layer(block, in_channels=channels[3], out_channels=channels[4], blocks=layers[3], resample='upsample')

        self.bn1 = torch.nn.BatchNorm2d(channels[-1])
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv1 = torch.nn.Conv2d(in_channels=channels[-1], out_channels=3, kernel_size=3, stride=1, padding=1)
        
    def make_layer(self, block, in_channels, out_channels, blocks, resample):
        layers = []
        layers.append(block(in_channels, out_channels, resample=resample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.dense(x)
        out = out.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv1(self.relu(self.bn1(out)))

        return out

class ResNetDiscriminator(torch.nn.Module):
    def __init__(self, block, channels, layers, input_shape, latent_dim, conditional=False, num_classes=10):
        super(ResNetDiscriminator, self).__init__()

        self.conditional = conditional
        self.input_shape = input_shape
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=3, stride=1, padding=1)
        self.layer1 = self.make_layer(block, in_channels=channels[0], out_channels=channels[1], blocks=layers[0], resample='downsample')
        self.layer2 = self.make_layer(block, in_channels=channels[1], out_channels=channels[2], blocks=layers[1], resample='downsample')
        self.layer3 = self.make_layer(block, in_channels=channels[2], out_channels=channels[3], blocks=layers[2], resample='downsample')
        self.layer4 = self.make_layer(block, in_channels=channels[3], out_channels=channels[4], blocks=layers[3], resample=None)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=(input_shape[1]//8, input_shape[2]//8))
        self.dense = Linear(in_features=channels[-1], out_features=latent_dim)
        if conditional:
            self.dense_cond = torch.nn.Linear(in_features=channels[-1], out_features=num_classes, bias=False)

        
    def make_layer(self, block, in_channels, out_channels, blocks, resample):
        layers = []
        layers.append(block(in_channels, out_channels, resample=resample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return torch.nn.Sequential(*layers)
    
    def forward(self, x, y=None):
        out = self.conv1(x)
        # out = out.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        if self.conditional:
            y_bar = self.dense_cond(out)
            similarity = (y * y_bar).sum(dim=1)
            out = self.dense(out) + similarity.view(-1, 1)
        else:
            out = self.dense(out)
        return out

class SoftTree(torch.nn.Module):
    
    def __init__(self, in_features, out_features, depth, projection=False, dropout=0.0):
        super(SoftTree, self).__init__()
        self.proj = projection
        self.depth = depth
        self.in_features = in_features
        self.out_features = out_features
        self.leaf_count = int(numpy.power(2,depth))
        self.gate_count = int(self.leaf_count - 1)
        self.gw = torch.nn.Parameter(
            torch.nn.init.kaiming_normal_(
                torch.empty(self.gate_count, in_features), nonlinearity='sigmoid').t())
        self.gb = torch.nn.Parameter(torch.zeros(self.gate_count))
        # dropout rate for gating weights.
        # see: Ahmetoglu et al. 2018 https://doi.org/10.1007/978-3-030-01418-6_14
        self.drop = torch.nn.Dropout(p=dropout)
        if self.proj:
            self.pw = torch.nn.init.kaiming_normal_(torch.empty(out_features*self.leaf_count, in_features), nonlinearity='linear')
            self.pw = torch.nn.Parameter(self.pw.view(out_features, self.leaf_count, in_features).permute(0, 2, 1))
            self.pb = torch.nn.Parameter(torch.zeros(out_features, self.leaf_count))
        else:
            # find a better init for this.
            self.z = torch.nn.Parameter(torch.randn(out_features, self.leaf_count))
        
    def forward(self,x):
        gw_ = self.drop(self.gw)
        gatings = torch.sigmoid(torch.add(torch.matmul(x,gw_),self.gb))
        leaf_probs = None
        for i in range(self.leaf_count):
            gateways = numpy.binary_repr(i,width=self.depth)
            index = 1
            probs = None
            for j in range(self.depth):
                if j == 0:
                    if gateways[j] == '0':
                        probs = gatings[:,index-1]
                        index = 2 * index
                    else:
                        probs = 1-gatings[:,index-1]
                        index = 2 * index + 1
                else:
                    if gateways[j] == '0':
                        probs = probs * gatings[:,index-1]
                        index = 2 * index
                    else:
                        probs = probs * (1-gatings[:,index-1])
                        index = 2 * index + 1
            if i == 0:
                leaf_probs = probs
            else:
                leaf_probs = torch.cat([leaf_probs,probs],dim=0)
        leaf_probs = leaf_probs.view(self.leaf_count,-1)
        if self.proj:
            gated_projection = torch.matmul(self.pw,leaf_probs).permute(2,0,1)
            gated_bias = torch.matmul(self.pb,leaf_probs).permute(1,0)
            result = torch.matmul(gated_projection,x.view(-1,self.in_features,1))[:,:,0] + gated_bias
        else:
            result = torch.matmul(self.z,leaf_probs).permute(1,0)
        return result
    
    def extra_repr(self):
        return "SoftTree(in_features=%d, out_features=%d, depth=%d, projection=%s)" % (
            self.in_features,
            self.out_features,
            self.depth,
            self.proj)
    
    def node_densities(self, x):
        with torch.no_grad():
            gw_ = self.drop(self.gw)
            gatings = torch.sigmoid(torch.add(torch.matmul(x,gw_),self.gb))
            node_densities = gatings[:,0].view(-1, 1)
            node_densities = torch.cat([node_densities, (1-gatings[:,0]).view(-1, 1)], dim=1)
            for i in range(1, self.gate_count):
                parent = i - 1
                parent_gating = node_densities[:, parent]
                current_left = torch.mul(parent_gating, gatings[:, i]).view(-1, 1)
                current_right = torch.mul(parent_gating, 1-gatings[:, i]).view(-1, 1)
                node_densities = torch.cat([node_densities, current_left, current_right], dim=1)
        return node_densities

class SoftTreeDecoder(torch.nn.Module):
    def __init__(self, channels, input_shape, latent_dim, depth, activation=torch.nn.ReLU(), dropout=0.0, projection=False, std=None, normalization=None):
        super(SoftTreeDecoder, self).__init__()
        self.input_shape = input_shape
        self.tree = SoftTree(in_features=latent_dim, out_features=input_shape[0] * input_shape[1] * input_shape[2], depth=depth, dropout=dropout, projection=projection)

        convolutions = []
        current_shape = input_shape
        for ch in channels[:-1]:
            convolutions.append(ConvTranspose2d(in_channels=current_shape[0], out_channels=ch, kernel_size=4, stride=2, padding=1, std=std))
            current_shape = [ch, current_shape[1] * 2, current_shape[2] * 2]
            if normalization == 'batch_norm':
                convolutions.append(torch.nn.BatchNorm2d(ch))
            elif normalization == 'layer_norm':
                convolutions.append(torch.nn.LayerNorm(current_shape))
            convolutions.append(activation)
        convolutions.append(ConvTranspose2d(in_channels=current_shape[0], out_channels=channels[-1], kernel_size=4, stride=2, padding=1, std=std))
        self.convolutions = torch.nn.Sequential(*convolutions)

    def forward(self, x):
        out = self.tree(x)
        out = out.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        out = self.convolutions(out)
        return out

class SoftTreeEncoder(torch.nn.Module):
    def __init__(self, channels, input_shape, latent_dim, depth, activation=torch.nn.ReLU(), dropout=0.0, projection=False, std=None, normalization=None):
        super(SoftTreeEncoder, self).__init__()
        convolutions = []
        current_shape = input_shape
        for ch in channels:
            convolutions.append(Conv2d(in_channels=current_shape[0], out_channels=ch, kernel_size=4, stride=2, padding=1, std=std))
            current_shape = [ch, current_shape[1] // 2, current_shape[2] // 2]
            if normalization == 'batch_norm':
                convolutions.append(torch.nn.BatchNorm2d(ch))
            elif normalization == 'layer_norm':
                convolutions.append(torch.nn.LayerNorm(current_shape))
            convolutions.append(activation)
        self.convolutions = torch.nn.Sequential(*convolutions)
        self.tree = SoftTree(in_features=current_shape[0] * current_shape[1] * current_shape[2], out_features=latent_dim, depth=depth, dropout=dropout, projection=projection)

    def forward(self, x):
        out = self.convolutions(x)
        out = out.view(out.shape[0], -1)
        out = self.tree(out)
        return out

'''a dummy identity function for copying a layer activation'''
class I(torch.nn.Module):
    def __init__(self):
        super(I, self).__init__()

    def forward(self, x):
        return x

    def extra_repr(self):
        return 'identity function'

'''last layer activations of InceptionV3 trained on ImageNet'''
class InceptionV3(torch.nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.inception = torchvision.models.inception_v3(pretrained=True)
        self.inception.eval()
        self.inception.fc = I()
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        if x.shape[1] != 3:
            x = x.repeat(1,3,1,1)
        x = (x-self.mean)/self.std
        x = self.transform(x, mode='bilinear', size=(299, 299), align_corners=False)
        return self.inception(x)