import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

import resnet3D

__all__ = [
    'PreActivationResNet3D', 'preact_resnet3d10', 'preact_resnet3d18', 'preact_resnet3d34',
    'preact_resnet3d50', 'preact_resnet3d101', 'preact_resnet3d152', 'preact_resnet3d200'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class PreActivationBasicBlock(nn.Module):
    expansion = 1
    Conv3d = staticmethod(conv3x3x3)

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActivationBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv1 = self.Conv3d(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = self.Conv3d(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreActivationBottleneck(nn.Module):
    expansion = 4
    Conv3d = nn.Conv3d

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActivationBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv1 = self.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = self.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes)
        self.conv3 = self.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreActivationResNet3D(resnet3D.ResNet3D):
    pass


def preact_resnet3d10(**kwargs):
    """Constructs a ResNet-10 model."""
    model = PreActivationResNet3D(PreActivationBasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def preact_resnet3d18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = PreActivationResNet3D(PreActivationBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def preact_resnet3d34(**kwargs):
    """Constpreact_ructs a ResNet-34 model."""
    model = PreActivationResNet3D(PreActivationBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def preact_resnet3d50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = PreActivationResNet3D(PreActivationBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def preact_resnet3d101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = PreActivationResNet3D(PreActivationBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def preact_resnet3d152(**kwargs):
    """Constructs a ResNet-101 model."""
    model = PreActivationResNet3D(PreActivationBottleneck, [3, 8, 36, 3], **kwargs)
    return model


def preact_resnet3d200(**kwargs):
    """Constructs a ResNet-101 model."""
    model = PreActivationResNet3D(PreActivationBottleneck, [3, 24, 36, 3], **kwargs)
    return model


if __name__ == '__main__':
    batch_size = 1
    num_frames = 8
    num_classes = 174
    img_feature_dim = 512
    frame_size = 224
    model = preact_resnet3d34(num_classes=num_classes)

    input_var = torch.autograd.Variable(torch.randn(batch_size, 3, num_frames, 224, 224))
    output = model(input_var)
    print(output.shape)

    model = preact_resnet3d50(num_classes=num_classes)

    input_var = torch.autograd.Variable(torch.randn(batch_size, 3, num_frames, 224, 224))
    output = model(input_var)
    print(output.shape)
