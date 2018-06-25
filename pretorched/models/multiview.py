import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _triple, _pair

import resnet3D


class MultiViewConv(nn.Conv2d):
    r"""TODO

    TODO

    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

        # If ints are given, convert them to an iterable, 1 -> [1, 1, 1].
        padding = _triple(padding)
        kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.dilation = _triple(dilation)
        self.channel_shape = (out_channels, in_channels // groups)

        self.paddings = [
            (0, padding[1], padding[2]),
            (padding[0], 0, padding[2]),
            (padding[0], padding[1], 0),
        ]

        self.kernel_sizes = [
            (1, kernel_size[1], kernel_size[2]),
            (kernel_size[0], 1, kernel_size[2]),
            (kernel_size[0], kernel_size[1], 1),
        ]
        self.linear = nn.Linear(3, 1)

    def forward(self, input):
        x = torch.stack([
            F.conv3d(input,
                     self.weight.view(*self.channel_shape, *kernel_size),
                     self.bias, self.stride, padding, self.dilation, self.groups)
            for kernel_size, padding in zip(self.kernel_sizes, self.paddings)], -1)
        x = self.linear(x)[..., 0]
        return x


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 MultiView convolution with padding."""
    return MultiViewConv(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(resnet3D.BasicBlock):
    Conv3d = staticmethod(conv3x3x3)


class Bottleneck(resnet3D.Bottleneck):
    Conv3d = MultiViewConv


class MVResNet(resnet3D.ResNet3D):

    Conv3d = MultiViewConv

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, MultiViewConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def mvresnet10(**kwargs):
    """Constructs a MVResNet-10 model.
    """
    model = MVResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def mvresnet18(**kwargs):
    """Constructs a MVResNet-18 model.
    """
    model = MVResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def mvresnet34(**kwargs):
    """Constructs a MVResNet-34 model.
    """
    model = MVResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def mvresnet50(**kwargs):
    """Constructs a MVResNet-50 model.
    """
    model = MVResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def mvresnet101(**kwargs):
    """Constructs a MVResNet-101 model.
    """
    model = MVResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def mvresnet152(**kwargs):
    """Constructs a MVResNet-101 model.
    """
    model = MVResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def mvresnet200(**kwargs):
    """Constructs a MVResNet-200 model.
    """
    model = MVResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


if __name__ == '__main__':
    batch_size = 1
    num_frames = 8
    num_classes = 174
    img_feature_dim = 512
    frame_size = 224
    model = mvresnet18(num_classes=num_classes)
    print(model)

    input_var = torch.autograd.Variable(torch.randn(batch_size, 3, num_frames, 224, 224))
    output = model(input_var)
    print(output.shape)

    input_var = torch.autograd.Variable(torch.randn(batch_size, 3, 12, 224, 224))
    output = model(input_var)
    print(output.shape)

    model = mvresnet50(num_classes=num_classes)
    print(model)

    input_var = torch.autograd.Variable(torch.randn(batch_size, 3, num_frames, 224, 224))
    output = model(input_var)
    print(output.shape)

    input_var = torch.autograd.Variable(torch.randn(batch_size, 3, 12, 224, 224))
    output = model(input_var)
    # print(output.shape)
    # print(model.state_dict().keys())

    # batch_size = 1
    # num_frames = 8
    # num_classes = 174
    # img_feature_dim = 512
    # frame_size = 224

    # input_var = torch.autograd.Variable(torch.randn(batch_size, 3, num_frames, frame_size, frame_size))
    # conv = MultiViewConv(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
    # out = conv(input_var)
    # print(out.shape)

    # x = []
    # print(f'input:  {input.squeeze().shape}')
    # for dim, (kernel_size, padding) in enumerate(zip(self.kernel_sizes, self.paddings)):
    # print(-(3 - dim))
    # print(f'kernel_size:  {kernel_size}')

    # weight = self.weight.view(self.out_channels, self.in_channels // self.groups, *kernel_size)
    # weight = self.weight.unsqueeze(-(dim + 1)).expand(self.out_channels, self.in_channels // self.groups, *kernel_size)
    # weight = self.weight.unsqueeze(-(3 - dim)).expand(self.out_channels, self.in_channels // self.groups, *kernel_size)
    # weight = self.weight.unsqueeze(-(3 - dim)).expand(self.out_channels, self.in_channels // self.groups, *kernel_size)
    # print(f'weight: {weight.shape}')
    # o = F.conv3d(input, weight, self.bias, self.stride, padding, self.dilation, self.groups)
    # print(f'output: {o.squeeze().shape}')
    # x.append(o)
    # x = torch.stack(x, -1)
    # x = self.linear(x)[..., 0]

    # x = self.linear(x)[...:0]
