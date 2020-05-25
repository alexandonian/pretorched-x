from collections import defaultdict
from typing import Dict, Union

import torch
import torch.nn as nn

from .torchvision_models import load_pretrained, modify_resnets

__all__ = [
    'resnet50x1',
    'resnet50x2',
    'resnet50x4',
]

model_urls: Dict[str, Dict[str, Union[str, None]]] = {
    'imagenet': defaultdict(
        lambda: None,
        {
            'resnet50x1': 'http://pretorched-x.csail.mit.edu/models/simclr_resnet50x1_imagenet-b9742372.pth',
            'resnet50x2': 'http://pretorched-x.csail.mit.edu/models/simclr_resnet50x2_imagenet-3264f065.pth',
            'resnet50x4': 'http://pretorched-x.csail.mit.edu/models/simclr_resnet50x4_imagenet-88b7e923.pth',
        },
    )
}

pretrained_settings: Dict[str, Dict[str, Dict]] = defaultdict(dict)
input_sizes = {}
means = {}
stds = {}
num_classes = {'imagenet': 1000}

for model_name in __all__:
    input_sizes[model_name] = [3, 1, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]

for model_name in __all__:
    if model_name in ['ResNet3D']:
        continue
    for dataset, urls in model_urls.items():
        pretrained_settings[model_name][dataset] = {
            'input_space': 'RGB',
            'input_range': [0, 1],
            'url': urls[model_name],
            'std': stds[model_name],
            'mean': means[model_name],
            'num_classes': num_classes[dataset],
            'input_size': input_sizes[model_name],
        }


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        self.downsample = (
            downsample  # hack: moving downsample to the first to make order correct
        )
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        width_mult=1,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64 * width_mult
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 * width_mult, layers[0])
        self.layer2 = self._make_layer(
            block,
            128 * width_mult,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256 * width_mult,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512 * width_mult,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion * width_mult, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet50x1(num_classes=1000, pretrained='imagenet', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, width_mult=1)
    if pretrained is not None:
        settings = pretrained_settings['resnet50x1'][pretrained]
        model = load_pretrained(model, num_classes, settings)
        print(f'Note: {pretrained} model was trained without normalization!!!')
    model = modify_resnets(model)
    return model


def resnet50x2(num_classes=1000, pretrained='imagenet', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, width_mult=2)
    if pretrained is not None:
        settings = pretrained_settings['resnet50x2'][pretrained]
        model = load_pretrained(model, num_classes, settings)
        print(f'Note: {pretrained} model was trained without normalization!!!')
    model = modify_resnets(model)
    return model


def resnet50x4(num_classes=1000, pretrained='imagenet', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, width_mult=4)
    if pretrained is not None:
        settings = pretrained_settings['resnet50x4'][pretrained]
        model = load_pretrained(model, num_classes, settings)
        print(f'Note: {pretrained} model was trained without normalization!!!')
    model = modify_resnets(model)
    return model
