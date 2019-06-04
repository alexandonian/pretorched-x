import torch
import torch.nn as nn


__all__ = ['resnet18', 'resnet50', 'resnet101', 'resnet152', 'resnet200']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 head_conv=1):
        super().__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes,
                                   kernel_size=(1, 3, 3),
                                   padding=(0, 1, 1),
                                   stride=(1, stride, stride),
                                   bias=False)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes,
                                   kernel_size=(3, 1, 1),
                                   padding=(1, 0, 0),
                                   bias=False)
        else:
            raise ValueError('Unsupported head_conv')

        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes,
                               kernel_size=(1, 3, 3),
                               padding=(0, 1, 1),
                               stride=(1, stride, stride))
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                                   bias=False, padding=(1, 0, 0))
        else:
            raise ValueError("Unsupported head_conv!")
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
            padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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


class Slow(nn.Module):

    def __init__(self, block=Bottleneck, layers=[2, 2, 2, 2]):
        super().__init__()
        self.inplanes = 64 + 64 // 8 * 2
        self._make_layers(block, layers)

    def _make_layers(self, block, layers):
        self.conv1 = nn.Conv3d(3, 64,
                               kernel_size=(1, 7, 7),
                               stride=(1, 2, 2),
                               padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                    stride=(1, 2, 2),
                                    padding=(0, 1, 1))

        self.res2 = self._make_layer_slow(block, 64, layers[0],
                                          head_conv=1)

        # TODO: Verify that this adjustment is correct.
        res3_stride = 2 if issubclass(block, Bottleneck) else 1
        self.res3 = self._make_layer_slow(block, 128, layers[1],
                                          stride=res3_stride,
                                          head_conv=1)

        self.res4 = self._make_layer_slow(block, 256, layers[2],
                                          stride=2,
                                          head_conv=3)

        self.res5 = self._make_layer_slow(block, 512, layers[3],
                                          stride=2,
                                          head_conv=3)

    def forward(self, input, lateral):

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.cat([x, lateral[0]], dim=1)
        x = self.res2(x)
        x = torch.cat([x, lateral[1]], dim=1)
        x = self.res3(x)
        x = torch.cat([x, lateral[2]], dim=1)
        x = self.res4(x)
        x = torch.cat([x, lateral[3]], dim=1)
        x = self.res5(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                head_conv=head_conv))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, head_conv=head_conv))

        self.inplanes = (
            planes * block.expansion + planes * block.expansion // 8 * 2)
        return nn.Sequential(*layers)


class SlowOnly(Slow):

    def __init__(self, block=Bottleneck, layers=[2, 2, 2, 2],
                 num_classes=400, dropout=0.5, slow_stride=16):
        nn.Module.__init__(self)
        self.inplanes = 64
        self.slow_stride = slow_stride
        self._make_layers(block, layers)
        self.dropout = nn.Dropout(dropout)
        self.last_linear = nn.Linear(self.inplanes, num_classes)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                head_conv=head_conv))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, head_conv=head_conv))

        self.inplanes = (
            planes * block.expansion)
        return nn.Sequential(*layers)

    def input_transform(self, input):
        return input[:, :, ::self.slow_stride, :, :]

    def forward(self, input):
        x = self.input_transform(input)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        x = self.dropout(x)
        x = self.last_linear(x)
        return x


class Fast(nn.Module):
    def __init__(self, block=Bottleneck, layers=[2, 2, 2, 2]):
        super().__init__()
        self.inplanes = 8
        self.conv1 = nn.Conv3d(3, 8,
                               kernel_size=(5, 7, 7),
                               stride=(1, 2, 2),
                               padding=(2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                    stride=(1, 2, 2),
                                    padding=(0, 1, 1))
        self.res2 = self._make_layer_fast(block, 8, layers[0],
                                          head_conv=3)
        res3_stride = 2 if issubclass(block, Bottleneck) else 1
        self.res3 = self._make_layer_fast(block, 16, layers[1],
                                          stride=res3_stride,
                                          head_conv=3)
        self.res4 = self._make_layer_fast(block, 32, layers[2],
                                          stride=2,
                                          head_conv=3)
        self.res5 = self._make_layer_fast(block, 64, layers[3],
                                          stride=2,
                                          head_conv=3)
        expansion = 4 if issubclass(block, Bottleneck) else 1
        self._make_lateral_layers(expansion)

    def _make_lateral_layers(self, expansion):
        self.lateral_p1 = nn.Conv3d(8, 8 * 2,
                                    kernel_size=(5, 1, 1),
                                    stride=(8, 1, 1),
                                    bias=False,
                                    padding=(2, 0, 0))
        r2 = 8 * expansion
        self.lateral_res2 = nn.Conv3d(r2, r2 * 2,
                                      kernel_size=(5, 1, 1),
                                      stride=(8, 1, 1),
                                      bias=False,
                                      padding=(2, 0, 0))
        r3 = 16 * expansion
        self.lateral_res3 = nn.Conv3d(r3, r3 * 2,
                                      kernel_size=(5, 1, 1),
                                      stride=(8, 1, 1),
                                      bias=False,
                                      padding=(2, 0, 0))
        r4 = 32 * expansion
        self.lateral_res4 = nn.Conv3d(r4, r4 * 2,
                                      kernel_size=(5, 1, 1),
                                      stride=(8, 1, 1),
                                      bias=False,
                                      padding=(2, 0, 0))

    def forward(self, input):
        lateral = []
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        pool1 = self.maxpool(x)
        lateral.append(self.lateral_p1(pool1))

        res2 = self.res2(pool1)
        lateral.append(self.lateral_res2(res2))

        res3 = self.res3(res2)
        lateral.append(self.lateral_res3(res3))

        res4 = self.res4(res3)
        lateral.append(self.lateral_res4(res4))

        res5 = self.res5(res4)
        x = nn.AdaptiveAvgPool3d(1)(res5)
        x = x.view(-1, x.size(1))
        return x, lateral

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            head_conv=head_conv))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)


class FastOnly(Fast):
    def __init__(self, block=Bottleneck, layers=[2, 2, 2, 2],
                 num_classes=400, dropout=0.5, fast_stride=2):
        super().__init__(block=block, layers=layers)
        self.fast_stride = fast_stride
        self.dropout = nn.Dropout(dropout)
        self.last_linear = nn.Linear(self.inplanes, num_classes)

    def _make_lateral_layers(self, expansion):
        return None

    def input_transform(self, input):
        return input[:, :, ::self.fast_stride, :, :]

    def forward(self, input):
        x = self.input_transform(input)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        pool1 = self.maxpool(x)
        res2 = self.res2(pool1)
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        res5 = self.res5(res4)
        x = nn.AdaptiveAvgPool3d(1)(res5)
        x = x.view(-1, x.size(1))
        x = self.dropout(x)
        x = self.last_linear(x)
        return x


class SlowFast(nn.Module):
    """SlowFast Network.

    Constructed from Slow and Fast nets.

    """

    def __init__(self, block=Bottleneck, layers=[2, 2, 2, 2], num_classes=400,
                 dropout=0.5, slow_stride=16, fast_stride=2):
        super().__init__()
        self.slow_stride = slow_stride
        self.fast_stride = fast_stride
        self.expansion = 4 if issubclass(block, Bottleneck) else 1
        self.slow = Slow(block=block, layers=layers)
        self.fast = Fast(block=block, layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.last_linear = nn.Linear(self.fast.inplanes + 512 * self.expansion,
                                     num_classes, bias=False)

    def forward(self, input):
        fast, lateral = self.fast(input[:, :, ::self.fast_stride, :, :])
        slow = self.slow(input[:, :, ::self.slow_stride, :, :], lateral)
        x = torch.cat([slow, fast], dim=1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x


class SlowFastV0(nn.Module):
    """Original SlowFast implementation where slow and fast pathways are built
    together.

    Advantages: Clean API.
    Disadvantages: Difficult to use individual pathways separately.

    """

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=10,
                 dropout=0.5):
        super().__init__()

        self.fast_inplanes = 8
        self.fast_conv1 = nn.Conv3d(3, 8,
                                    kernel_size=(5, 7, 7),
                                    stride=(1, 2, 2),
                                    padding=(2, 3, 3),
                                    bias=False)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                         stride=(1, 2, 2),
                                         padding=(0, 1, 1))
        self.fast_res2 = self._make_layer_fast(block, 8, layers[0],
                                               head_conv=3)
        self.fast_res3 = self._make_layer_fast(block, 16, layers[1],
                                               stride=2,
                                               head_conv=3)
        self.fast_res4 = self._make_layer_fast(block, 32, layers[2],
                                               stride=2,
                                               head_conv=3)
        self.fast_res5 = self._make_layer_fast(block, 64, layers[3],
                                               stride=2,
                                               head_conv=3)

        self.lateral_p1 = nn.Conv3d(8, 8 * 2,
                                    kernel_size=(5, 1, 1),
                                    stride=(8, 1, 1),
                                    bias=False,
                                    padding=(2, 0, 0))
        self.lateral_res2 = nn.Conv3d(32, 32 * 2,
                                      kernel_size=(5, 1, 1),
                                      stride=(8, 1, 1),
                                      bias=False,
                                      padding=(2, 0, 0))
        self.lateral_res3 = nn.Conv3d(64, 64 * 2,
                                      kernel_size=(5, 1, 1),
                                      stride=(8, 1, 1),
                                      bias=False,
                                      padding=(2, 0, 0))
        self.lateral_res4 = nn.Conv3d(128, 128 * 2,
                                      kernel_size=(5, 1, 1),
                                      stride=(8, 1, 1),
                                      bias=False,
                                      padding=(2, 0, 0))

        self.slow_inplanes = 64 + 64 // 8 * 2
        self.slow_conv1 = nn.Conv3d(3, 64,
                                    kernel_size=(1, 7, 7),
                                    stride=(1, 2, 2),
                                    padding=(0, 3, 3),
                                    bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                         stride=(1, 2, 2),
                                         padding=(0, 1, 1))
        self.slow_res2 = self._make_layer_slow(block, 64, layers[0],
                                               head_conv=1)
        self.slow_res3 = self._make_layer_slow(block, 128, layers[1],
                                               stride=2,
                                               head_conv=1)
        self.slow_res4 = self._make_layer_slow(block, 256, layers[2],
                                               stride=2,
                                               head_conv=3)
        self.slow_res5 = self._make_layer_slow(block, 512, layers[3],
                                               stride=2,
                                               head_conv=3)
        self.dropout = nn.Dropout(dropout)
        self.last_linear = nn.Linear(self.fast_inplanes + 2048, num_classes,
                                     bias=False)

    def forward(self, input):
        fast, lateral = self.fast_path(input[:, :, ::2, :, :])
        slow = self.slow_path(input[:, :, ::16, :, :], lateral)
        x = torch.cat([slow, fast], dim=1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def slow_path(self, input, lateral):
        x = self.slow_conv1(input)
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        x = torch.cat([x, lateral[0]], dim=1)
        x = self.slow_res2(x)
        x = torch.cat([x, lateral[1]], dim=1)
        x = self.slow_res3(x)
        x = torch.cat([x, lateral[2]], dim=1)
        x = self.slow_res4(x)
        x = torch.cat([x, lateral[3]], dim=1)
        x = self.slow_res5(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x

    def fast_path(self, input):
        lateral = []
        x = self.fast_conv1(input)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x)
        lateral_p = self.lateral_p1(pool1)
        lateral.append(lateral_p)

        res2 = self.fast_res2(pool1)
        lateral_res2 = self.lateral_res2(res2)
        lateral.append(lateral_res2)

        res3 = self.fast_res3(res2)
        lateral_res3 = self.lateral_res3(res3)
        lateral.append(lateral_res3)

        res4 = self.fast_res4(res3)
        lateral_res4 = self.lateral_res4(res4)
        lateral.append(lateral_res4)

        res5 = self.fast_res5(res4)
        x = nn.AdaptiveAvgPool3d(1)(res5)
        x = x.view(-1, x.size(1))

        return x, lateral

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample,
                            head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes,
                                head_conv=head_conv))
        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.slow_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.slow_inplanes, planes, stride, downsample,
                            head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes,
                                head_conv=head_conv))

        self.slow_inplanes = (
            planes * block.expansion + planes * block.expansion // 8 * 2)
        return nn.Sequential(*layers)


def resnet18(mode='SF', **kwargs):
    """Constructs a ResNet-18 model.
    """
    models = {'sf': SlowFast, 'f': FastOnly, 's': SlowOnly}
    Func = models.get(mode.lower(), 'sf')
    return Func(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(mode='SF', **kwargs):
    """Constructs a ResNet-50 model.
    """
    models = {'sf': SlowFast, 'f': FastOnly, 's': SlowOnly}
    Func = models.get(mode.lower(), 'sf')
    return Func(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


if __name__ == "__main__":
    num_classes = 12
    input_tensor = torch.rand(4, 3, 64, 224, 224)

    for mode in ['f', 's', 'sf']:
        out = resnet50(mode=mode, num_classes=num_classes)(input_tensor)
        print(f'resnet50 mode: {mode}', out.shape)
        out = resnet18(mode=mode, num_classes=num_classes)(input_tensor)
        print(f'resnet18 mode: {mode}', out.shape)
