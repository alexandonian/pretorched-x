import math
from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = [
    'ResNeXt3D', 'resnext3d10', 'resnext3d18', 'resnext3d34',
    'resnext3d50', 'resnext3d101', 'resnext3d152', 'resnext3d200'
]

model_urls = {
    'kinetics-400': defaultdict(lambda: None, {
        'resnext3d101': 'http://pretorched-x.csail.mit.edu/models/resnext3d101_kinetics-8e57b772.pth',
    }),
}

num_classes = {
    'kinetics-400': 400,
    'moments': 339,
}

pretrained_settings = {}
input_sizes = {}
means = {}
stds = {}

for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]

for model_name in __all__:
    for dataset, urls in model_urls.items():
        pretrained_settings[model_name] = {
            dataset: {
                'input_space': 'RGB',
                'input_range': [0, 1],
                'url': urls[model_name],
                'std': stds[model_name],
                'mean': means[model_name],
                'num_classes': num_classes[dataset],
                'input_size': input_sizes[model_name],
            }
        }


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt3D(nn.Module):

    def __init__(self, block, layers, shortcut_type='B', cardinality=32, num_classes=400):
        self.inplanes = 64
        super(ResNeXt3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnext3d10(**kwargs):
    """Constructs a ResNeXt3D-10 model."""
    model = ResNeXt3D(ResNeXtBottleneck, [1, 1, 1, 1], **kwargs)
    return model


def resnext3d18(**kwargs):
    """Constructs a ResNeXt3D-18 model."""
    model = ResNeXt3D(ResNeXtBottleneck, [2, 2, 2, 2], **kwargs)
    return model


def resnext3d34(**kwargs):
    """Constructs a ResNeXt3D-34 model."""
    model = ResNeXt3D(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext3d50(**kwargs):
    """Constructs a ResNeXt3D-50 model."""
    model = ResNeXt3D(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext3d101(**kwargs):
    """Constructs a ResNeXt3D-101 model."""
    model = ResNeXt3D(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext3d152(**kwargs):
    """Constructs a ResNeXt3D-152 model."""
    model = ResNeXt3D(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnext3d200(**kwargs):
    """Constructs a ResNeXt3D-200 model."""
    model = ResNeXt3D(ResNeXtBottleneck, [3, 24, 36, 3], **kwargs)
    return model
