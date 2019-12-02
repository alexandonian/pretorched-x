import torch
import torch.nn as nn
from torch.nn import init

import pretorched.layers as layers


class GBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size, stride,
                                       padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(True)

    def forward(self, input):
        """Forward method of GBasicBlock.

        This block increases the spatial resolution by 2:

            input:  [batch_size, in_channels, H, W]
            output: [batch_size, out_channels, 2*H, 2*W]
        """

        x = self.conv(input)
        x = self.bn(x)
        out = self.act(x)
        return out


class DBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """Forward method of DBlock.

        This block decreases the spatial resolution by 2:

            input:  [batch_size, in_channels, H, W]
            output: [batch_size, out_channels, H/2, W/2]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_features,
                 batchnorm_func=layers.ConditionalBatchNorm2d):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn = batchnorm_func(out_channels, num_features)
        self.act = nn.ReLU(True)

    def forward(self, x, y=None):
        x = self.conv(x)
        if y is not None:
            x = self.bn(x, y)
        else:
            x = self.bn(x)
        x = self.bn(x) if y is None else self.bn(x, y)
        x = self.act(x)
        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm_func=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn = batchnorm_func(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """Forward method of DBlock.

        This block decreases the spatial resolution by 2:

            input:  [batch_size, in_channels, H, W]
            output: [batch_size, out_channels, H/2, W/2]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Generator(nn.Module):
    """DCGAN Generator."""

    # Maps output resoluton to number of GBlocks.
    res2blocks = {32: 3, 64: 4, 128: 5, 256: 6, 512: 7}

    def __init__(self, dim_z=100, resolution=64, G_ch=64, block=GBasicBlock, init='N02'):
        super().__init__()

        self.G_ch = G_ch
        self.init = init
        self.dim_z = dim_z

        self.num_blocks = self.res2blocks[resolution]
        self.ch_dims = [G_ch * (2**i) for i in range(self.num_blocks, 0, -1)]

        self.input = block(dim_z, self.ch_dims[0], kernel_size=4, stride=1, padding=0)
        self.GBlocks = nn.Sequential(*[
            block(in_c, out_c) for in_c, out_c in zip(self.ch_dims, self.ch_dims[1:])
        ])

        self.out = nn.ConvTranspose2d(self.ch_dims[-1], 3, 4, 2, 1)  # RGB image has 3 channels
        self.tanh = nn.Tanh()                                        # "Squashes" out to be in range[-1, 1]

        self.init_weights()

    def init_weights(self):
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

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        x = self.input(x)
        x = self.GBlocks(x)
        x = self.out(x)
        return self.tanh(x)


class ConditionalGenerator(nn.Module):
    res2blocks = {
        32: 3,
        64: 4,
        128: 5,
        256: 6,
    }

    def __init__(self, dim_z=128, n_classes=2, resolution=128, G_ch=64, shared_dim=128,
                 block_func=GBlock, init='N02'):
        super().__init__()

        self.G_ch = G_ch
        self.init = init
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.shared_dim = shared_dim

        self.num_blocks = self.res2blocks[resolution]
        self.fnums = [2**i for i in range(self.num_blocks)]
        self.fnums += self.fnums[-1:]
        self.fnums = list(reversed(self.fnums))

        self.shared = nn.Embedding(n_classes, shared_dim)
        self.linear = nn.Linear(dim_z, G_ch * self.fnums[0] * 4**2)

        self.GBlocks = nn.ModuleList([
            block_func(G_ch * in_c, G_ch * out_c, shared_dim)
            for in_c, out_c in zip(self.fnums, self.fnums[1:])])

        self.out = nn.Conv2d(G_ch * 1, 3, 3, padding=1)
        self.tanh = nn.Tanh()

        self.init_weights()

    def forward(self, z, y):
        class_embed = self.shared(y)
        return self.generate(z, class_embed)

    def generate(self, z, class_embed):
        z = self.linear(z).view(z.size(0), -1, 4, 4)
        for block in self.GBlocks:
            z = block(z, class_embed)
        return self.tanh(self.out(z))

    def init_weights(self):
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


####################################################################
# DCGAN Discriminator
####################################################################


class _Discriminator(nn.Module):
    """DCGAN discriminator."""

    # Maps output resoluton to number of DBlocks.
    res2blocks = {
        32: 3,
        64: 4,
        128: 5,
        256: 6,
    }

    def __init__(self, resolution=128, D_ch=64, block=DBlock):
        super().__init__()
        self.D_ch = D_ch
        self.num_blocks = self.res2blocks[resolution]
        self.ch_dims = [2**i for i in range(self.num_blocks)]
        self.input = nn.Conv2d(3, D_ch, 3, padding=1)

        self.DBlocks = nn.Sequential(*[
            block(D_ch * in_c, D_ch * out_c)
            for in_c, out_c in zip(self.ch_dims, self.ch_dims[1:])
        ])

        self.out = nn.Conv2d(D_ch * self.ch_dims[-1], 1, 3, 1, 0)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.DBlocks(x)
        x = self.act(torch.mean(self.out(x), [2, 3]))
        return x


class Discriminator(nn.Module):
    """DCGAN discriminator."""

    # Maps output resoluton to number of DBlocks.
    res2blocks = {32: 3, 64: 4, 128: 5, 256: 6, 512: 7}

    def __init__(self, resolution=128, D_ch=64, block=DBlock, n_classes=1000):
        super().__init__()
        self.D_ch = D_ch
        self.n_classess = n_classes
        self.num_blocks = self.res2blocks[resolution]
        self.ch_dims = [D_ch * (2**i) for i in range(self.num_blocks)]
        self.input = nn.Sequential(
            nn.Conv2d(3, self.ch_dims[0], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.DBlocks = nn.Sequential(*[
            block(in_c, out_c) for in_c, out_c in zip(self.ch_dims, self.ch_dims[1:])
        ])

        self.out = nn.Conv2d(self.ch_dims[-1], 1, 4, 1, 0)
        self.act = nn.Sigmoid()

    def forward(self, x, y=None):
        x = self.input(x)
        x = self.DBlocks(x)
        x = self.out(x)
        x = self.act(x)
        return x.view(-1)


class ProjectionDiscriminator(Discriminator):
    """cGAN Projection discriminator."""

    # Maps output resoluton to number of DBlocks.
    res2blocks = {
        32: 3,
        64: 4,
        128: 5,
        256: 6,
    }

    def __init__(self, n_classes=1000, resolution=128, D_ch=64, block=DBlock):
        super().__init__(resolution=resolution, D_ch=D_ch, block=block)

        del self.out
        self.embed = nn.Embedding(n_classes, self.ch_dims[-1])
        self.linear = nn.Linear(self.ch_dims[-1], 1)
        self.act = nn.ReLU()

    def forward(self, x, y=None):
        h = x
        h = self.input(h)
        h = self.DBlocks(h)
        h = self.act(h)

        # Apply global sum pooling.
        h = h.sum([2, 3])

        # Initial class-unconditional output.
        out = self.linear(h)

        # Get projection of final featureset onto class vectors and add to evidence.
        if y is not None:
            out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        return out.view(-1)
