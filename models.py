import numpy
import math
import torch
import torchvision
import utils

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
            if self.conv.bias is not None:
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
            if self.convt.bias is not None:
                self.convt.bias.data.normal_(0., std)

    def forward(self, x):
        x = self.convt(x)
        return x
    
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, std=None, bias=True, normalization=None, transposed=False):
        super(ConvBlock, self).__init__()
        if transposed:
            self.block = [torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        else:
            self.block = [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        self.block.append(torch.nn.ReLU())
        if normalization == "batch_norm":
            self.block.append(torch.nn.BatchNorm2d(out_channels))

        if std is not None:
            self.block[0].weight.data.normal_(0., std)
            self.block[0].bias.data.normal_(0., std)
        self.block = torch.nn.Sequential(*self.block)
    
    def forward(self, x):
        return self.block(x)

'''DCGAN-like convolutional encoder'''
class ConvEncoder(torch.nn.Module):
    def __init__(self, channels, input_shape, latent_dim, activation=torch.nn.ReLU(), std=None, normalization=None, conditional=False, num_classes=10, parallel=False):
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
        if parallel:
            self.convolutions = torch.nn.DataParallel(self.convolutions)
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

class ConvEncoder2(torch.nn.Module):
    def __init__(self, channels, input_shape, latent_dim, activation=torch.nn.ReLU(), std=None, normalization=None):
        super(ConvEncoder2, self).__init__()
        convolutions = []
        current_shape = input_shape
        for ch in channels:
            convolutions.append(Conv2d(in_channels=current_shape[0], out_channels=ch, kernel_size=4, stride=2, padding=1, std=std))
            convolutions.append(activation)
            current_shape = [ch, current_shape[1] // 2, current_shape[2] // 2]
            if normalization == 'batch_norm':
                convolutions.append(torch.nn.BatchNorm2d(ch))
            elif normalization == 'layer_norm':
                convolutions.append(torch.nn.LayerNorm(current_shape))
        self.convolutions = torch.nn.Sequential(*convolutions)
        self.dense = Linear(in_features=current_shape[0] * current_shape[1] * current_shape[2], out_features=latent_dim, std=std)

    def forward(self, x):
        out = self.convolutions(x)
        out = out.view(out.shape[0], -1)
        out = self.dense(out)
        return out

class MGANDisc(torch.nn.Module):
    def __init__(self, channels, input_shape, num_g, activation=torch.nn.ReLU(), std=None, normalization=None):
        super(MGANDisc, self).__init__()
        convolutions = []
        current_shape = input_shape
        for ch in channels:
            convolutions.append(Conv2d(in_channels=current_shape[0], out_channels=ch, kernel_size=4, stride=2, padding=1, std=std))
            convolutions.append(activation)
            current_shape = [ch, current_shape[1] // 2, current_shape[2] // 2]
            if normalization == 'batch_norm':
                convolutions.append(torch.nn.BatchNorm2d(ch))
            elif normalization == 'layer_norm':
                convolutions.append(torch.nn.LayerNorm(current_shape))
        self.convolutions = torch.nn.Sequential(*convolutions)
        self.discriminator = Linear(in_features=current_shape[0] * current_shape[1] * current_shape[2], out_features=1, std=std)
        self.classifier = Linear(in_features=current_shape[0] * current_shape[1] * current_shape[2], out_features=num_g, std=std)

    def forward(self, x):
        out = self.convolutions(x)
        out = out.view(out.shape[0], -1)
        d = self.discriminator(out)
        c = self.classifier(out)
        return d, c 


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

class SoftTree(torch.nn.Module):
    
    def __init__(self, in_features, out_features, depth, projection='constant', dropout=0.0):
        super(SoftTree, self).__init__()
        self.proj = projection
        self.depth = depth
        self.in_features = in_features
        self.out_features = out_features
        self.leaf_count = int(2**depth)
        self.gate_count = int(self.leaf_count - 1)
        self.gw = torch.nn.Parameter(
            torch.nn.init.kaiming_normal_(
                torch.empty(self.gate_count, in_features), nonlinearity='sigmoid').t())
        self.gb = torch.nn.Parameter(torch.zeros(self.gate_count))
        # dropout rate for gating weights.
        self.drop = torch.nn.Dropout(p=dropout)
        if self.proj == 'linear':
            self.pw = torch.nn.init.kaiming_normal_(torch.empty(out_features*self.leaf_count, in_features), nonlinearity='linear')
            self.pw = torch.nn.Parameter(self.pw.view(out_features, self.leaf_count, in_features).permute(0, 2, 1))
            self.pb = torch.nn.Parameter(torch.zeros(out_features, self.leaf_count))
        elif self.proj == 'linear2':
            self.pw = torch.nn.init.kaiming_normal_(torch.empty(out_features*self.leaf_count, in_features), nonlinearity='linear')
            # [leaf, in, out]
            self.pw = torch.nn.Parameter(self.pw.view(out_features, self.leaf_count, in_features).permute(1, 2, 0))
            # [leaf, 1, out]
            self.pb = torch.nn.Parameter(torch.zeros(self.leaf_count, 1, out_features))
        elif self.proj == 'constant':
            # find a better init for this.
            self.z = torch.nn.Parameter(torch.randn(out_features, self.leaf_count))
        elif self.proj == 'gmm':
            self.mu = torch.nn.Parameter(torch.randn(out_features, self.leaf_count))
            self.std = torch.nn.Parameter(torch.randn(out_features, self.leaf_count))
        
    def forward(self, x):
        node_densities = self.node_densities(x)
        leaf_probs = node_densities[:, -self.leaf_count:].t()

        if self.proj == 'linear':
            gated_projection = torch.matmul(self.pw,leaf_probs).permute(2,0,1)
            gated_bias = torch.matmul(self.pb,leaf_probs).permute(1,0)
            result = torch.matmul(gated_projection,x.view(-1,self.in_features,1))[:,:,0] + gated_bias
        elif self.proj == 'linear2':
            # input = [1, batch, dim]
            x = x.view(1, x.shape[0], x.shape[1])
            out = torch.matmul(x, self.pw)+self.pb
            result = (out, leaf_probs)
        elif self.proj == 'constant':
            result = torch.matmul(self.z,leaf_probs).permute(1,0)
        elif self.proj == 'gmm':
            mu = self.mu.view(1, self.out_features, self.leaf_count)
            std = self.std.view(1, self.out_features, self.leaf_count)
            eps = torch.randn(x.shape[0], self.out_features, self.leaf_count, device=x.device)
            z = mu + std * eps
            leaf_probs = leaf_probs.t().view(-1, self.leaf_count, 1)
            result = torch.bmm(z, leaf_probs)[:, :, 0]
        return result
    
    def extra_repr(self):
        return "in_features=%d, out_features=%d, depth=%d, projection=%s" % (
            self.in_features,
            self.out_features,
            self.depth,
            self.proj)
    
    def node_densities(self, x):
        gw_ = self.drop(self.gw)
        gatings = torch.sigmoid(torch.add(torch.matmul(x,gw_),self.gb))
        node_densities = torch.ones(x.shape[0], 2**(self.depth+1)-1, device=x.device)
        it = 1
        for d in range(1, self.depth+1):
            for i in range(2**d):
                parent_index = (it+1) // 2 - 1
                child_way = (it+1) % 2
                if child_way == 0:
                    parent_gating = gatings[:, parent_index]
                else:
                    parent_gating = 1 - gatings[:, parent_index]
                parent_density = node_densities[:, parent_index].clone()
                node_densities[:, it] = (parent_density * parent_gating)
                it += 1
        return node_densities
    
    def gatings(self, x):
        return torch.sigmoid(torch.add(torch.matmul(x,self.gw),self.gb))
    
    def total_path_value(self, z, index, level=None):
        gatings = self.gatings(z)
        gateways = numpy.binary_repr(index, width=self.depth)
        L = 0.
        current = 0
        if level is None:
            level = self.depth

        for i in range(level):
            if int(gateways[i]) == 0:
                L += gatings[:, current].mean()
                current = 2 * current + 1
            else:
                L += (1 - gatings[:, current]).mean()
                current = 2 * current + 2
        return L

class MixtureDecoder(torch.nn.Module):
    def __init__(self, channels, input_shape, latent_dim, mixture, depth, projection='constant', activation=torch.nn.ReLU(), dropout=0.0, std=None, normalization=None):
        super(MixtureDecoder, self).__init__()
        self.input_shape = input_shape
        if mixture == 'tree':
            self.mixture = SoftTree(in_features=latent_dim, out_features=input_shape[0] * input_shape[1] * input_shape[2], depth=depth, dropout=dropout, projection=projection)
        else:
            self.mixture = MoE(in_features=latent_dim, out_features=input_shape[0] * input_shape[1] * input_shape[2], num_leafs=2**depth, dropout=dropout, projection=projection)

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
        out = self.mixture(x)
        out = out.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        out = self.convolutions(out)
        return out

class MixtureEncoder(torch.nn.Module):
    def __init__(self, channels, input_shape, latent_dim, mixture, depth, projection='constant', activation=torch.nn.ReLU(), dropout=0.0, std=None, normalization=None):
        super(MixtureEncoder, self).__init__()
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
        if mixture == 'tree':
            self.mixture = SoftTree(in_features=current_shape[0] * current_shape[1] * current_shape[2], out_features=latent_dim, depth=depth, dropout=dropout, projection=projection)
        else:
            self.mixture = MoE(in_features=current_shape[0] * current_shape[1] * current_shape[2], out_features=latent_dim, num_leafs=2**depth, dropout=dropout, projection=projection)

    def forward(self, x):
        out = self.convolutions(x)
        out = out.view(out.shape[0], -1)
        out = self.mixture(out)
        return out

class MoE(torch.nn.Module):
    
    def __init__(self, in_features, out_features, num_leafs, projection='constant', dropout=0.0):
        super(MoE, self).__init__()
        self.proj = projection
        self.num_leafs = num_leafs
        self.in_features = in_features
        self.out_features = out_features
        self.gw = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(
                torch.empty(in_features, num_leafs)))
        self.gb = torch.nn.Parameter(torch.zeros(num_leafs))
        # dropout rate for gating weights.
        # see: Ahmetoglu et al. 2018 https://doi.org/10.1007/978-3-030-01418-6_14
        self.drop = torch.nn.Dropout(p=dropout)
        if self.proj == 'linear':
            self.pw = torch.nn.init.kaiming_normal_(torch.empty(out_features*num_leafs, in_features), nonlinearity='linear')
            self.pw = torch.nn.Parameter(self.pw.view(out_features, num_leafs, in_features).permute(0, 2, 1))
            self.pb = torch.nn.Parameter(torch.zeros(out_features, num_leafs))
        elif self.proj == 'constant':
            # find a better init for this.
            self.z = torch.nn.Parameter(torch.randn(out_features, num_leafs))
        elif self.proj == 'gmm':
            self.mu = torch.nn.Parameter(torch.randn(out_features, num_leafs))
            self.std = torch.nn.Parameter(torch.randn(out_features, num_leafs))
        
    def forward(self, x):
        gw_ = self.drop(self.gw)
        gatings = torch.softmax(torch.add(torch.matmul(x,gw_),self.gb), dim=1).t()
        if self.proj == 'linear':
            gated_projection = torch.matmul(self.pw, gatings).permute(2,0,1)
            gated_bias = torch.matmul(self.pb, gatings).permute(1,0)
            result = torch.matmul(gated_projection,x.view(-1,self.in_features,1))[:,:,0] + gated_bias
        elif self.proj == 'constant':
            result = torch.matmul(self.z, gatings).permute(1,0)
        elif self.proj == 'gmm':
            mu = self.mu.view(1, self.out_features, self.num_leafs)
            std = self.std.view(1, self.out_features, self.num_leafs)
            eps = torch.randn(x.shape[0], self.out_features, self.num_leafs, device=x.device)
            z = mu + std * eps
            gatings = gatings.t().view(-1, self.num_leafs, 1)
            result = torch.bmm(z, gatings)[:, :, 0]
        return result
    
    def extra_repr(self):
        return "in_features=%d, out_features=%d, num_leafs=%d, projection=%s" % (
            self.in_features,
            self.out_features,
            self.num_leafs,
            self.proj)

class MADGAN(torch.nn.Module):
    def __init__(self, num_of_generators, channels, input_shape, latent_dim, std=None, normalization=None):
        super(MADGAN, self).__init__()
        self.num_of_generators = num_of_generators
        projected_dim = input_shape[0]*input_shape[1]*input_shape[2]
        self.weight = torch.nn.init.kaiming_normal_(torch.empty(num_of_generators * projected_dim, latent_dim), nonlinearity='relu')
        self.weight = torch.nn.Parameter(self.weight.view(num_of_generators, projected_dim, latent_dim).permute(0, 2, 1))
        self.bias = torch.nn.Parameter(torch.zeros(num_of_generators, 1, projected_dim))
        self.bn = torch.nn.BatchNorm1d(num_of_generators * projected_dim)
                
        self.shared_block = [
                ConvBlock(
                    in_channels=input_shape[0],
                    out_channels=channels[0],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    std=std,
                    normalization=normalization,
                    transposed=True
                    )
                ]
        for ch in range(len(channels)-2):
            self.shared_block.append(
                ConvBlock(
                    in_channels=channels[ch],
                    out_channels=channels[ch+1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    std=std,
                    normalization=normalization,
                    transposed=True
                    )
                )
        self.shared_block.append(
                ConvTranspose2d(
                    in_channels=channels[-2],
                    out_channels=channels[-1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    std=std
                    )
                )
        self.shared_block = torch.nn.Sequential(*self.shared_block)

    def forward(self, x):
        # (num_g, batch, dim)
        h = torch.relu(torch.matmul(x, self.weight)+self.bias)
        # (batch, num_g, dim)
        h = h.permute(1,0,2).contiguous()
        # (batch, num_g * dim)
        h = self.bn(h.view(x.shape[0], -1))
        # (batch, num_g, dim)
        h = h.view(x.shape[0], self.num_of_generators, -1)
        # (num_g, batch, dim)
        h = h.permute(1,0,2).contiguous()
        # (num_g * batch, channel, height, width)
        h = h.view(x.shape[0] * self.num_of_generators, -1, 4, 4)

        out = self.shared_block(h) 
        return out

class MEGANGen(torch.nn.Module):
    def __init__(self, num_of_generators, channels, input_shape, latent_dim, std=None, normalization=None):
        super(MEGANGen, self).__init__()
        self.num_of_generators = num_of_generators
        self.normalization = normalization
        projected_dim = input_shape[0]*input_shape[1]*input_shape[2]
        self.weight = torch.nn.init.kaiming_normal_(torch.empty(num_of_generators * projected_dim, latent_dim), nonlinearity='relu')
        self.weight = torch.nn.Parameter(self.weight.view(num_of_generators, projected_dim, latent_dim).permute(0, 2, 1))
        self.bias = torch.nn.Parameter(torch.zeros(num_of_generators, 1, projected_dim))
        if normalization == "batch_norm":
            self.bn = torch.nn.BatchNorm1d(num_of_generators * projected_dim)
        self.feat_projector = Linear(in_features=projected_dim, out_features=latent_dim)
        self.gating = Linear(in_features=latent_dim*(num_of_generators+1), out_features=num_of_generators)

        
        self.shared_block = [
                ConvBlock(
                    in_channels=input_shape[0],
                    out_channels=channels[0],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    std=std,
                    normalization=normalization,
                    transposed=True
                    )
                ]
        for ch in range(len(channels)-2):
            self.shared_block.append(
                ConvBlock(
                    in_channels=channels[ch],
                    out_channels=channels[ch+1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    std=std,
                    normalization=normalization,
                    transposed=True
                    )
                )
        self.shared_block.append(
                ConvTranspose2d(
                    in_channels=channels[-2],
                    out_channels=channels[-1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    std=std
                    )
                )
        self.shared_block = torch.nn.Sequential(*self.shared_block)

    def forward(self, x):
        # (num_g, batch, dim)
        h = torch.relu(torch.matmul(x, self.weight)+self.bias)
        # (batch, num_g, dim)
        h = h.permute(1,0,2).contiguous()
        # (batch, num_g * dim)
        if self.normalization is not None:
            h = self.bn(h.view(x.shape[0], -1))
        else:
            h = h.view(x.shape[0], -1)
        # (batch, num_g, dim)
        h = h.view(x.shape[0], self.num_of_generators, -1)
        
        # calculate gate logits 
        feats = self.feat_projector(h.view(x.shape[0]*self.num_of_generators, -1))
        feats = feats.view(x.shape[0], self.num_of_generators, -1)
        feats = torch.cat([feats, x.unsqueeze(1)], dim=1).view(x.shape[0], -1)
        gate_logits = self.gating(feats) 

        # gating
        gate = utils.gumbel_softmax(gate_logits).unsqueeze(2)
        h = torch.mul(h, gate).sum(dim=1)
         
        ## sample generation
        # (batch, channel, height, width)
        h = h.view(x.shape[0], -1, 4, 4)

        out = self.shared_block(h)
        # out = out.view(x.shape[0], self.num_of_generators, out.shape[1], out.shape[2], out.shape[3])
        return out, gate_logits

class MultiLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, num_of_generators):
        super(MultiLinear, self).__init__()
        self.num_of_generators = num_of_generators
        self.generators = []
        for i in range(num_of_generators):
            self.generators.append(MLP([in_features, out_features, out_features], activation=torch.nn.Tanh()))
        self.generators = torch.nn.ModuleList(self.generators)
        
    def forward(self, x):
        o = []
        for i in range(self.num_of_generators):
            o.append(self.generators[i](x))
        return torch.cat(o, dim=0)

class MultiMEGAN(torch.nn.Module):
    def __init__(self, in_features, out_features, num_of_generators):
        super(MultiMEGAN, self).__init__()
        self.num_of_generators = num_of_generators
        self.generators = []
        for i in range(num_of_generators):
            self.generators.append(MLP([in_features, out_features, out_features], activation=torch.nn.Tanh()))
        self.generators = torch.nn.ModuleList(self.generators)
        self.gating = torch.nn.Linear(in_features=in_features*(num_of_generators+1), out_features=num_of_generators)
        self.feat_projector = Linear(in_features=out_features, out_features=in_features)

    def forward(self, x):
        o = []
        for i in range(self.num_of_generators):
            o.append(self.generators[i](x))
        o = torch.cat(o, dim=0).view(self.num_of_generators, x.shape[0], -1)
        
        # (batch, num_g, dim)
        o = o.permute(1,0,2).contiguous()
        feat = self.feat_projector(o.view(x.shape[0] * self.num_of_generators, -1)).view(x.shape[0], self.num_of_generators, -1)
        feat = torch.cat([feat, x.unsqueeze(1)], dim=1).view(x.shape[0], -1)
        gating_logits = self.gating(feat)
        gate = utils.gumbel_softmax(gating_logits).unsqueeze(2)
        o = torch.mul(o, gate).sum(dim=1)
        return o, gating_logits

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

def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

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
    def __init__(self, in_channels, out_channels, normalization, stride=1, resample=None, input_size=None):
        super(PreActResidualBlock, self).__init__()
        self.block = []
        if normalization == "batch_norm":
            self.block.append(torch.nn.BatchNorm2d(in_channels))
        elif normalization == "layer_norm":
            self.block.append(torch.nn.LayerNorm((in_channels, input_size[0], input_size[1])))            
        self.block.append(torch.nn.ReLU(inplace=True))
        self.block.append(conv3x3(in_channels, out_channels, stride))
        if normalization == "batch_norm":
            self.block.append(torch.nn.BatchNorm2d(out_channels))
        elif normalization == "layer_norm":
            self.block.append(torch.nn.LayerNorm((out_channels, input_size[0], input_size[1])))
        self.block.append(torch.nn.ReLU(inplace=True))
        self.block.append(conv3x3(out_channels, out_channels))
        self.resample = resample
        if in_channels > out_channels:
            self.conv1x1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.block = torch.nn.Sequential(*self.block)

    def forward(self, x):
        residual = x
        out = self.block(x)
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

class ResNetGenerator(torch.nn.Module):
    def __init__(self, block, channels, layers, input_shape, latent_dim, depth, out_channels, normalization, projection="linear", dropout=0.0, parallel=False):
        super(ResNetGenerator, self).__init__()

        self.input_shape = input_shape
        self.dense = SoftTree(in_features=latent_dim, out_features=input_shape[0] * input_shape[1] * input_shape[2], depth=depth, dropout=dropout, projection=projection)
        self.res_blocks = []
        for i in range(len(layers)):
            self.res_blocks.append(self.make_layer(block, in_channels=channels[i], out_channels=channels[i+1], blocks=layers[i], resample="upsample", normalization=normalization, input_size=None))
        
        if normalization == "batch_norm":
            self.res_blocks.append(torch.nn.BatchNorm2d(channels[-1]))
        self.res_blocks.append(torch.nn.ReLU(inplace=True))
        self.res_blocks.append(torch.nn.Conv2d(in_channels=channels[-1], out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        self.res_blocks = torch.nn.Sequential(*self.res_blocks)
        if parallel:
            self.res_blocks = torch.nn.DataParallel(self.res_blocks)
        
    def make_layer(self, block, in_channels, out_channels, blocks, resample, normalization, input_size):
        layers = []
        layers.append(block(in_channels, out_channels, resample=resample, normalization=normalization, input_size=input_size))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, normalization=normalization))
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.dense(x)
        out = out.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        return self.res_blocks(out)

class ResNetDiscriminator(torch.nn.Module):
    def __init__(self, block, channels, layers, input_shape, latent_dim, in_channels, normalization, parallel=False):
        super(ResNetDiscriminator, self).__init__()
        scale_factor = 2**(len(layers)-1)
        self.res_blocks = []
        self.res_blocks.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=channels[0], kernel_size=3, stride=1, padding=1))
        current_shape = input_shape[1:].copy()
        for i in range(len(layers)-1):
            self.res_blocks.append(self.make_layer(block, in_channels=channels[i], out_channels=channels[i+1], blocks=layers[i], resample="downsample", normalization=normalization, input_size=current_shape))
            current_shape[0] //= 2
            current_shape[1] //= 2
        self.res_blocks.append(self.make_layer(block, in_channels=channels[-2], out_channels=channels[-1], blocks=layers[-1], resample=None, normalization=normalization, input_size=current_shape))
        self.res_blocks.append(torch.nn.AvgPool2d(kernel_size=(input_shape[1]//scale_factor, input_shape[2]//scale_factor)))
        self.res_blocks = torch.nn.Sequential(*self.res_blocks)
        if parallel:
            self.res_blocks = torch.nn.DataParallel(self.res_blocks)
        
        self.dense = Linear(in_features=channels[-1], out_features=latent_dim)
        
    def make_layer(self, block, in_channels, out_channels, blocks, resample, normalization, input_size):
        layers = []
        layers.append(block(in_channels, out_channels, resample=resample, normalization=normalization, input_size=input_size))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.res_blocks(x)
        out = out.view(out.shape[0], -1)
        out = self.dense(out)
        return out
