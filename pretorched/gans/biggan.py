"""biggan.py

Implements BigGAN architecture from

Large Scale GAN Training for High Fidelity Natural Image Synthesis by Andrew Brock, Jeff Donahue, and Karen Simonyan.

Adapted from https://github.com/ajbrock/BigGAN-PyTorch.

"""
import functools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

import pretorched.layers as layers


class GBlock(nn.Module):
    """Residual block for generator.

    Note that this class assumes the kernel size and padding (and any other
    settings) have been selected in the main generator module and passed in
    through the conv_func arg. Similar rules apply with bn_func (the input
    size [which is actually the number of channels of the conditional info] must
    be preselected)

    """

    def __init__(self, in_channels, out_channels,
                 conv_func=nn.Conv2d,
                 bn_func=layers.bn,
                 activation=None,
                 upsample=None):
        super().__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv_func, self.bn_func = conv_func, bn_func
        self.activation = activation
        self.upsample = upsample

        # Conv layers
        self.conv1 = self.conv_func(self.in_channels, self.out_channels)
        self.conv2 = self.conv_func(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.conv_func(in_channels, out_channels,
                                          kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.bn_func(in_channels)
        self.bn2 = self.bn_func(out_channels)

    def forward(self, x, y):
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class DBlock(nn.Module):
    """ Residual block for discriminator."""

    def __init__(self, in_channels, out_channels, conv_func=layers.SNConv2d, wide=True,
                 preactivation=False, activation=None, downsample=None,):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels

        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.conv_func = conv_func
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.conv_func(self.in_channels, self.hidden_channels)
        self.conv2 = self.conv_func(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.conv_func(in_channels, out_channels,
                                          kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            h = F.relu(x)  # This line *must* be an out-of-place ReLU or it will negatively affect the shortcut connection.
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)


def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    """Architectures for G.
    Attention is passed in in the format '32_64' to mean applying an attention
    block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64
    """
    arch = {}
    arch[512] = {'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
                 'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1, 1]],
                 'upsample': [True] * 7,
                 'resolution': [8, 16, 32, 64, 128, 256, 512],
                 'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                               for i in range(3, 10)}}
    arch[256] = {'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1]],
                 'upsample': [True] * 6,
                 'resolution': [8, 16, 32, 64, 128, 256],
                 'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                               for i in range(3, 9)}}
    arch[128] = {'in_channels': [ch * item for item in [16, 16, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 4, 2, 1]],
                 'upsample': [True] * 5,
                 'resolution': [8, 16, 32, 64, 128],
                 'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                               for i in range(3, 8)}}
    arch[64] = {'in_channels': [ch * item for item in [16, 16, 8, 4]],
                'out_channels': [ch * item for item in [16, 8, 4, 2]],
                'upsample': [True] * 4,
                'resolution': [8, 16, 32, 64],
                'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3, 7)}}
    arch[32] = {'in_channels': [ch * item for item in [4, 4, 4]],
                'out_channels': [ch * item for item in [4, 4, 4]],
                'upsample': [True] * 3,
                'resolution': [8, 16, 32],
                'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3, 6)}}

    return arch


class Generator(nn.Module):
    def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
                 G_kernel_size=3, G_attn='64', n_classes=1000,
                 num_G_SVs=1, num_G_SV_itrs=1,
                 G_shared=True, shared_dim=0, hier=False,
                 cross_replica=False, mybn=False,
                 G_activation=nn.ReLU(inplace=False),
                 G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
                 BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
                 G_init='ortho', skip_init=False, no_optim=False,
                 G_param='SN', norm_style='bn', verbose=False,
                 **kwargs):
        super(Generator, self).__init__()
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.dim_z = dim_z
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_classes
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Print out model information during init?
        self.verbose = verbose
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[resolution]

        # If using hierarchical latents, adjust z
        if self.hier:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.dim_z // self.num_slots)
            # Recalculate latent dimensionality for even splitting into chunks
            self.dim_z = self.z_chunk_size * self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.conv_func = functools.partial(layers.SNConv2d,
                                               kernel_size=3, padding=1,
                                               num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                               eps=self.SN_eps)
            self.linear_func = functools.partial(layers.SNLinear,
                                                 num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                 eps=self.SN_eps)
        else:
            self.conv_func = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.linear_func = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        self.embedding_func = nn.Embedding
        bn_linear = (functools.partial(self.linear_func, bias=False) if self.G_shared
                     else self.embedding_func)
        self.bn_func = functools.partial(layers.ccbn,
                                         linear_func=bn_linear,
                                         cross_replica=self.cross_replica,
                                         mybn=self.mybn,
                                         input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                                     else self.n_classes),
                                         norm_style=self.norm_style,
                                         eps=self.BN_eps)

        # Prepare model
        # If not using shared embeddings, self.shared is just a passthrough
        self.shared = (self.embedding_func(n_classes, self.shared_dim) if G_shared
                       else nn.Identity())
        # First linear layer
        self.linear = self.linear_func(self.dim_z // self.num_slots,
                                       self.arch['in_channels'][0] * (self.bottom_width ** 2))

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[GBlock(in_channels=self.arch['in_channels'][index],
                                    out_channels=self.arch['out_channels'][index],
                                    conv_func=self.conv_func,
                                    bn_func=self.bn_func,
                                    activation=self.activation,
                                    upsample=(functools.partial(F.interpolate, scale_factor=2)
                                              if self.arch['upsample'][index] else None))]]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                if self.verbose:
                    print('Adding attention layer in G at resolution {}'.format(self.arch['resolution'][index]))
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.conv_func)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn),
                                          self.activation,
                                          self.conv_func(self.arch['out_channels'][-1], 3))

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        # If this is an EMA copy, no need for an optim, so just return now
        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            if self.verbose:
                print('Using fp16 adam in G...')
            from . import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0,
                                      eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0,
                                    eps=self.adam_eps)

        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        if self.verbose:
            print('Param count for G''s initialized parameters: {}'.format(self.param_count))

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, y, embed=False):
        if embed:
            y = self.shared(y)
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, ys[index])

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))

    @property
    def z_dim(self):
        return self.dim_z


def D_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    """Discriminator architecture, same paradigm as G's above."""
    arch = {}
    arch[256] = {'in_channels': [3] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[128] = {'in_channels': [3] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[64] = {'in_channels': [3] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[32] = {'in_channels': [3] + [item * ch for item in [4, 4, 4]],
                'out_channels': [item * ch for item in [4, 4, 4, 4]],
                'downsample': [True, True, False, False],
                'resolution': [16, 16, 16, 16],
                'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2, 6)}}
    return arch


class Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                 D_kernel_size=3, D_attn='64', n_classes=1000,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                 SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
                 D_init='ortho', skip_init=False, D_param='SN', verbose=False, **kwargs):
        super().__init__()
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Print model info during init?
        self.verbose = verbose
        # Architecture
        self.arch = D_arch(self.ch, self.attention)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.conv_func = functools.partial(layers.SNConv2d,
                                               kernel_size=3, padding=1,
                                               num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                               eps=self.SN_eps)
            self.linear_func = functools.partial(layers.SNLinear,
                                                 num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                 eps=self.SN_eps)
            self.embedding_func = functools.partial(layers.SNEmbedding,
                                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                    out_channels=self.arch['out_channels'][index],
                                    conv_func=self.conv_func,
                                    wide=self.D_wide,
                                    activation=self.activation,
                                    preactivation=(index > 0),
                                    downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                if self.verbose:
                    print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                     self.conv_func)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.linear_func(self.arch['out_channels'][-1], output_dim)
        # Embedding for projection discrimination
        self.embed = self.embedding_func(self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            if self.verbose:
                print('Using fp16 adam in D...')
            from . import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        if self.verbose:
            print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        out = self.linear(h)
        # Get projection of final featureset onto class vectors and add to evidence
        out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        return out


class G_D(nn.Module):
    """Parallelized G_D to minimize cross-gpu communication.
    Without this, Generator outputs would get all-gathered and then rebroadcast.
    """

    def __init__(self, G, D):
        super().__init__()
        self.G = G
        self.D = D

    def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
                split_D=False):
        # If training G, enable grad tape
        with torch.set_grad_enabled(train_G):
            # Get Generator output given noise
            G_z = self.G(z, self.G.shared(gy))
            # Cast as necessary
            if self.G.fp16 and not self.D.fp16:
                G_z = G_z.float()
            if self.D.fp16 and not self.G.fp16:
                G_z = G_z.half()
        # Split_D means to run D once with real data and once with fake,
        # rather than concatenating along the batch dimension.
        if split_D:
            D_fake = self.D(G_z, gy)
            if x is not None:
                D_real = self.D(x, dy)
                return D_fake, D_real
            else:
                if return_G_z:
                    return D_fake, G_z
                else:
                    return D_fake
        # If real data is provided, concatenate it with the Generator's output
        # along the batch dimension for improved efficiency.
        else:
            D_input = torch.cat([G_z, x], 0) if x is not None else G_z
            D_class = torch.cat([gy, dy], 0) if dy is not None else gy
            # Get Discriminator output
            D_out = self.D(D_input, D_class)
            if x is not None:
                return torch.split(D_out, [G_z.shape[0], x.shape[0]])  # D_fake, D_real
            else:
                if return_G_z:
                    return D_out, G_z
                else:
                    return D_out


root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
model_weights = {
    'imagenet': {
        128: {
            'G': os.path.join(root, '100k', 'G.pth'),
            'G_ema': os.path.join(root, '100k', 'G_ema.pth'),
            'state_dict': os.path.join(root, '100k', 'state_dict.pth'),
        }
    }
}

root_url = 'http://pretorched-x.csail.mit.edu/gans/BigGAN'
ganocracy_root_url = 'http://ganocracy.csail.mit.edu/models'
tfhub_urls = {
    'imagenet': {
        128: os.path.join(ganocracy_root_url, 'tfbiggan_128-13f17ff2.pth'),
        256: os.path.join(ganocracy_root_url, 'tfbiggan_256-a4cf3382.pth'),
        512: os.path.join(ganocracy_root_url, 'tfbiggan_512-447bfb81.pth'),
    }
}

model_urls = {
    'places365': {
        128: {
            'D': os.path.join(root_url, 'biggan128_D_places365-8afb2a4d.pth'),
            'G': os.path.join(root_url, 'biggan128_G_places365-43cd58c0.pth'),
            'G_ema': os.path.join(root_url, 'biggan128_G_ema_places365-78c78abe.pth'),
            'state_dict': os.path.join(root_url, 'biggan128_state_dict_places365-3d39f6bb.pth')
        },
        256: {
            'D': os.path.join(root_url, 'biggan256_D_places365-44bf2902.pth'),
            'G': os.path.join(root_url, 'biggan256_G_places365-5adac787.pth'),
            'G_ema': os.path.join(root_url, 'biggan256_G_ema_places365-ac277771.pth'),
            'state_dict': os.path.join(root_url, 'biggan256_state_dict_places365-b4f6daf6.pth'),
        },
    },
    'imagenet': {
        128: {
            'D': os.path.join(root_url, 'biggan128_D_imagenet-9fd72e50.pth'),
            'G': os.path.join(root_url, 'biggan128_G_imagenet-94e0b761.pth'),
            'G_ema': os.path.join(root_url, 'biggan128_G_ema_imagenet-c9706dfb.pth'),
            'state_dict': os.path.join(root_url, 'biggan128_state_dict_imagenet-4aad5089.pth'),
        },
    }
}


def BigGAN(resolution=256, pretrained='imagenet', load_ema=True, tfhub=True):

    attn = {128: '64', 256: '128', 512: '64'}
    dim_z = {128: 120, 256: 140, 512: 128}
    config = {
        'G_param': 'SN', 'D_param': 'SN',
        'G_ch': 96, 'D_ch': 96,
        'D_wide': True, 'G_shared': True,
        'shared_dim': 128, 'dim_z': dim_z[resolution],
        'hier': True, 'cross_replica': False,
        'mybn': False, 'G_activation': nn.ReLU(inplace=True),
        'G_attn': attn[resolution],
        'norm_style': 'bn',
        'G_init': 'ortho', 'skip_init': True, 'no_optim': True,
        'G_fp16': False, 'G_mixed_precision': False,
        'accumulate_stats': False, 'num_standing_accumulations': 16,
        'G_eval_mode': True,
        'BN_eps': 1e-04, 'SN_eps': 1e-04,
        'num_G_SVs': 1, 'num_G_SV_itrs': 1, 'resolution': resolution,
        'n_classes': 1000
    }

    version = 'G_ema' if load_ema else 'G'

    if tfhub and pretrained == 'imagenet':
        url = tfhub_urls[pretrained][resolution]
        weights = torch.hub.load_state_dict_from_url(url)
        G = Generator(**config)
        G.load_state_dict(weights, strict=False)
        G.eval()
        return G
    elif pretrained is not None:
        url = model_urls[pretrained][resolution][version]
        sd_url = model_urls[pretrained][resolution]['state_dict']
        weights = torch.hub.load_state_dict_from_url(url)
        state_dict = torch.hub.load_state_dict_from_url(sd_url)
        G = Generator(**state_dict['config'])
        G.load_state_dict(weights, strict=False)
        G.eval()
        return G
    G = Generator(**config)
    return G


def fix_class(G, y):
    f = G.forward

    def forward(self, z):
        bs = z.size(0)
        c = y * torch.ones(bs, device=z.device).long()
        return f(z, c, embed=True)

    setattr(G.__class__, 'forward', forward)
    return G
