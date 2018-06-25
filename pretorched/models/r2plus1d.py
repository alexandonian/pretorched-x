import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _triple

import resnet3D

__all__ = [
    'R2Plus1D', 'r2plus1d10', 'r2plus1d18', 'r2plus1d34', 'r2plus1d50', 'r2plus1d101',
    'r2plus1d152', 'r2plus1d200'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 Factored Spatial-Temporal convolution with padding."""
    return SpatioTemporalConv(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal.

    The input signal is composed of several input planes with distinct
    spatial and time axes,by performing a 2D convolution over the spatial
    axes to an intermediate subspace, followed by a 1D convolution over the
    time axis to produce the final output.

    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        # If ints are given, convert them to an iterable, 1 -> [1, 1, 1].
        stride = _triple(stride)
        padding = _triple(padding)
        kernel_size = _triple(kernel_size)

        # Decompose the parameters into spatial and temporal components
        # by masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid aberrant
        # behavior such as padding being added twice.
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]

        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]
        temporal_kernel_size = [kernel_size[0], 1, 1]

        # Compute the number of intermediary channels (M) using formula
        # from the paper section 3.5:
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) /
                                           (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # The spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU.
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # The temporal conv is effectively a 1D conv, but has batch norm
        # and ReLU added inside the model constructor, not here. This is an
        # intentional design choice, to allow this module to externally act
        # identically to a standard Conv3D, so it can be reused easily in any other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x


class BasicBlock(resnet3D.BasicBlock):
    Conv3d = staticmethod(conv3x3x3)


class Bottleneck(resnet3D.Bottleneck):
    Conv3d = SpatioTemporalConv


class R2Plus1D(resnet3D.ResNet3D):

    Conv3d = SpatioTemporalConv

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, SpatioTemporalConv):
                nn.init.kaiming_normal_(m.spatial_conv.weight, mode='fan_out')
                nn.init.kaiming_normal_(m.temporal_conv.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def r2plus1d10(**kwargs):
    """Constructs a ResNet-18 model."""
    model = R2Plus1D(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def r2plus1d18(**kwargs):
    """Constructs a R2Plus1D-18 model."""
    model = R2Plus1D(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def r2plus1d34(**kwargs):
    """Constructs a R2Plus1D-34 model."""
    model = R2Plus1D(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def r2plus1d50(**kwargs):
    """Constructs a R2Plus1D-50 model."""
    model = R2Plus1D(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def r2plus1d101(**kwargs):
    """Constructs a R2Plus1D-101 model."""
    model = R2Plus1D(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def r2plus1d152(**kwargs):
    """Constructs a R2Plus1D-101 model."""
    model = R2Plus1D(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def r2plus1d200(**kwargs):
    """Constructs a R2Plus1D-200 model."""
    model = R2Plus1D(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


if __name__ == '__main__':
    batch_size = 1
    num_frames = 8
    num_classes = 174
    img_feature_dim = 512
    frame_size = 224
    model = r2plus1d18(num_classes=num_classes)
    print(model)

    input_var = torch.autograd.Variable(torch.randn(batch_size, 3, num_frames, 224, 224))
    output = model(input_var)
    print(output.shape)

    model = r2plus1d50(num_classes=num_classes)
    print(model)

    input_var = torch.autograd.Variable(torch.randn(batch_size, 3, num_frames, 224, 224))
    output = model(input_var)
    print(output.shape)

    # input_var = torch.autograd.Variable(torch.randn(batch_size, 3, 12, 224, 224))
    # output = model(input_var)
    # print(output.shape)

    # model = r2plus1d50(num_classes=num_classes)

    # input_var = torch.autograd.Variable(torch.randn(batch_size, 3, num_frames, 224, 224))
    # output = model(input_var)
    # print(output.shape)

    # input_var = torch.autograd.Variable(torch.randn(batch_size, 3, 12, 224, 224))
    # output = model(input_var)
    # print(output.shape)
