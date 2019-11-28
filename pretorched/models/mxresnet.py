# FastAI's XResnet modified to use Mish activation function, MXResNet
# https://github.com/fastai/fastai/blob/master/fastai/vision/models/xresnet.py
# modified by lessw2020 - github:  https://github.com/lessw2020/mish


import math
import sys
from functools import partial

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.utils.spectral_norm import spectral_norm

from ..nn import Mish

__all__ = ['MXResNet', 'mxresnet18', 'mxresnet34', 'mxresnet50', 'mxresnet101', 'mxresnet152']


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    """Create and initialize a `nn.Conv1d` layer with spectral normalization.

    Unmodified from
    https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
    """
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    return spectral_norm(conv)


class SimpleSelfAttention(nn.Module):
    # Adapted from SelfAttention layer at https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
    # Inspired by https://arxiv.org/pdf/1805.08318.pdf

    def __init__(self, n_in: int, ks=1, sym=False):  # , n_out:int):
        super().__init__()

        self.conv = conv1d(n_in, n_in, ks, padding=ks // 2, bias=False)

        self.gamma = nn.Parameter(torch.tensor([0.]))

        self.sym = sym
        self.n_in = n_in

    def forward(self, x):

        if self.sym:
            # symmetry hack by https://github.com/mgrankin
            c = self.conv.weight.view(self.n_in, self.n_in)
            c = (c + c.t()) / 2
            self.conv.weight = c.view(self.n_in, self.n_in, 1)

        size = x.size()
        x = x.view(*size[:2], -1)   # (C,N)

        # changed the order of mutiplication to avoid O(N^2) complexity
        # (x*xT)*(W*x) instead of (x*(xT*(W*x)))

        convx = self.conv(x)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(x, x.permute(0, 2, 1).contiguous())   # (C,N) * (N,C) = (C,C)   => O(NC^2)

        o = torch.bmm(xxT, convx)   # (C,C) * (C,N) = (C,N)   => O(NC^2)

        o = self.gamma * o + x

        return o.view(*size).contiguous()


# or: ELU+init (a=0.54; gain=1.55)
act_fn = Mish()  # nn.ReLU(inplace=True)


def init_cnn(m):
    if getattr(m, 'bias', None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for l in m.children():
        init_cnn(l)


def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks // 2, bias=bias)


def noop(x):
    return x


def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm2d(nf)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, nf, ks, stride=stride), bn]
    if act:
        layers.append(act_fn)
    return nn.Sequential(*layers)


class ResBlock(Module):
    def __init__(self, expansion, ni, nh, stride=1, sa=False, sym=False):
        super().__init__()
        nf, ni = nh * expansion, ni * expansion
        layers = [conv_layer(ni, nh, 3, stride=stride),
                  conv_layer(nh, nf, 3, zero_bn=True, act=False)
                  ] if expansion == 1 else [
            conv_layer(ni, nh, 1),
            conv_layer(nh, nh, 3, stride=stride),
            conv_layer(nh, nf, 1, zero_bn=True, act=False)
        ]
        self.sa = SimpleSelfAttention(nf, ks=1, sym=sym) if sa else noop
        self.convs = nn.Sequential(*layers)
        # TODO: check whether act=True works better
        self.idconv = noop if ni == nf else conv_layer(ni, nf, 1, act=False)
        self.pool = noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return act_fn(self.sa(self.convs(x)) + self.idconv(self.pool(x)))


def filt_sz(recep):
    return min(64, 2**math.floor(math.log2(recep * 0.75)))


class MXResNetSeq(nn.Sequential):
    def __init__(self, expansion, layers, c_in=3, num_classes=1000, sa=False, sym=False):
        stem = []
        sizes = [c_in, 32, 64, 64]  # modified per Grankin
        for i in range(3):
            stem.append(conv_layer(sizes[i], sizes[i + 1], stride=2 if i == 0 else 1))

        block_szs = [64 // expansion, 64, 128, 256, 512]
        blocks = [self._make_layer(expansion, block_szs[i],
                                   block_szs[i + 1], l,
                                   1 if i == 0 else 2,
                                   sa=sa if i in[len(layers) - 4] else False,
                                   sym=sym)
                  for i, l in enumerate(layers)]
        super().__init__(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *blocks,
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(block_szs[-1] * expansion, num_classes),
        )
        init_cnn(self)

    def _make_layer(self, expansion, ni, nf, blocks, stride, sa=False, sym=False):
        return nn.Sequential(
            *[ResBlock(expansion, ni if i == 0 else nf, nf, stride if i == 0 else 1, sa if i in [blocks - 1] else False, sym)
              for i in range(blocks)])


class MXResNet(nn.Module):
    def __init__(self, expansion, layers, c_in=3, num_classes=1000, sa=False, sym=False):
        super().__init__()
        stem = []
        sizes = [c_in, 32, 64, 64]  # modified per Grankin
        for i in range(3):
            stem.append(conv_layer(sizes[i], sizes[i + 1], stride=2 if i == 0 else 1))

        block_szs = [64 // expansion, 64, 128, 256, 512]
        blocks = [self._make_layer(expansion, block_szs[i],
                                   block_szs[i + 1], l,
                                   1 if i == 0 else 2,
                                   sa=sa if i in[len(layers) - 4] else False,
                                   sym=sym)
                  for i, l in enumerate(layers)]
        self.last_linear = nn.Linear(block_szs[-1] * expansion, num_classes)
        self.features = nn.Sequential(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *blocks)
        self.logits = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            self.last_linear)

        init_cnn(self)

    def _make_layer(self, expansion, ni, nf, blocks, stride, sa=False, sym=False):
        return nn.Sequential(
            *[ResBlock(expansion, ni if i == 0 else nf, nf, stride if i == 0 else 1, sa if i in [blocks - 1] else False, sym)
              for i in range(blocks)])

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def mxresnet(expansion, n_layers, name, pretrained=False, **kwargs):
    model = MXResNet(expansion, n_layers, **kwargs)
    model.input_size = (3, 224, 224)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls[name]))
        print("No pretrained yet for MXResNet")
    return model


me = sys.modules[__name__]
for n, e, l in [
    [18, 1, [2, 2, 2, 2]],
    [34, 1, [3, 4, 6, 3]],
    [50, 4, [3, 4, 6, 3]],
    [101, 4, [3, 4, 23, 3]],
    [152, 4, [3, 8, 36, 3]],
]:
    name = f'mxresnet{n}'
    setattr(me, name, partial(mxresnet, expansion=e, n_layers=l, name=name))