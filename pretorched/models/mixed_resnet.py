import math
from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from .torchvision_models import load_pretrained, modify_resnets
from .resnet3D import (BasicBlock as BasicBlock3D, Bottleneck as Bottleneck3D, ResNet3D)
from .fbresnet import BasicBlock, Bottleneck, FBResNet

__all__ = [
    'ResNet3D', 'resnet3d10', 'resnet3d18', 'resnet3d34',
    'resnet3d50', 'resnet3d101', 'resnet3d152', 'resnet3d200',
]

model_urls = {
    'kinetics-400': defaultdict(lambda: None, {
        'resnet3d18': 'http://pretorched-x.csail.mit.edu/models/resnet3d18_kinetics-b5673e4a.pth',
        'resnet3d34': 'http://pretorched-x.csail.mit.edu/models/resnet3d34_kinetics-133ec9c4.pth',
        'resnet3d50': 'http://pretorched-x.csail.mit.edu/models/resnet3d50_kinetics-aad059c9.pth',
        'resnet3d101': 'http://pretorched-x.csail.mit.edu/models/resnet3d101_kinetics-a6ddd22a.pth',
        'resnet3d152': 'http://pretorched-x.csail.mit.edu/models/resnet3d152_kinetics-8ae08d3f.pth',
    }),
    'moments': defaultdict(lambda: None, {
        'resnet3d50': 'http://pretorched-x.csail.mit.edu/models/resnet3d50_16seg_moments-22f4fe61.pth',
    }),
}

num_classes = {'kinetics-400': 400, 'moments': 339}

pretrained_settings = defaultdict(dict)
input_sizes = {}
means = {}
stds = {}

for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
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


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1),
        out.size(2), out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNetMixed(nn.Module):

    Conv2d = nn.Conv2d
    Conv3d = nn.Conv3d

    def __init__(self, block, layers, model_type='top_heavy', shortcut_type='B', num_classes=339):
        if block == 'BasicBlock':
            block2d = BasicBlock
            block3d = BasicBlock3D
        elif block == 'Bottleneck':
            block2d = Bottleneck
            block3d = Bottleneck3D

        self.inplanes = 64
        self.model_type = model_type
        super().__init__()
        if self.model_type == 'top_heavy':
            self.conv1 = self.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = FBResNet._make_layer(self, block2d, 64, layers[0])
            self.layer2 = FBResNet._make_layer(self, block2d, 128, layers[1], stride=2)
            self.layer3 = ResNet3D._make_layer(self, block3d, 256, layers[2], shortcut_type, stride=2)
            self.layer4 = ResNet3D._make_layer(self, block3d, 512, layers[3], shortcut_type, stride=2)
            self.avgpool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Linear(512 * block3d.expansion, num_classes)

        elif self.model_type == 'bottom_heavy':
            self.conv1 = self.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
            self.bn1 = nn.BatchNorm3d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

            self.layer1 = self._make_layer(block3d, 64, layers[0], shortcut_type)
            self.layer2 = self._make_layer(block3d, 128, layers[1], shortcut_type, stride=2)
            self.layer3 = FBResNet._make_layer(self, block2d, 256, layers[2],  stride=2)
            self.layer4 = FBResNet._make_layer(self, block2d, 512, layers[3],  stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512 * block2d.expansion, num_classes)
        else:
            raise ValueError('Unrecognized model type.')

        self.init_weights()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    self.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, self.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        if self.model_type == 'top_heavy':
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(-1, 3, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        print(x.shape)
        # x = x.view(batch_size, -1, *x.shape[1:])
        if self.model_type == 'bottom_heavy':
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            print(x.shape)
            x = x.view(-1, *x.shape[-3:])

        else:
            x = x.view(batch_size, -1, *x.shape[1:])
            print(x.shape)
            x = x.permute(0, 2, 1, 3, 4)
        print(x.shape)
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


def resnet3d10(**kwargs):
    """Constructs a ResNet3D-10 model."""
    model = ResNet3D(BasicBlock, [1, 1, 1, 1], **kwargs)
    model = modify_resnets(model)
    return model


def resnet3d18(num_classes=400, pretrained='kinetics-400', shortcut_type='A', **kwargs):
    """Constructs a ResNet3D-18 model."""
    model = ResNet3D(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                     shortcut_type=shortcut_type, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet3d18'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet3d34(num_classes=400, pretrained='kinetics-400', shortcut_type='A', **kwargs):
    """Constructs a ResNet3D-34 model."""
    model = ResNet3D(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
                     shortcut_type=shortcut_type, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet3d34'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet3d50(num_classes=400, pretrained='kinetics-400', **kwargs):
    """Constructs a ResNet3D-50 model."""
    model = ResNetMixed('Bottleneck', [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet3d50'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    # model = modify_resnets(model)
    return model


def resnet3d101(num_classes=400, pretrained='kinetics-400', **kwargs):
    """Constructs a ResNet3D-101 model."""
    model = ResNet3D(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet3d101'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet3d152(num_classes=400, pretrained='kinetics-400', **kwargs):
    """Constructs a ResNet3D-152 model."""
    model = ResNet3D(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet3d152'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet3d200(num_classes=400, pretrained='kinetics-400', **kwargs):
    """Constructs a ResNet3D-200 model."""
    model = ResNet3D(Bottleneck, [3, 24, 36, 3], **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet3d200'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


if __name__ == '__main__':
    batch_size = 1
    num_frames = 48
    num_classes = 339
    img_feature_dim = 512
    frame_size = 224
    model = resnet3d50(num_classes=num_classes, pretrained='moments')

    input_var = torch.randn(batch_size, 3, num_frames, 224, 224)
    print(input_var.shape)
    output = model(input_var)
    print(output.shape)
