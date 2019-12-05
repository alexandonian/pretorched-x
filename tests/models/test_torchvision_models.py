from pretorched import models
import torch
import pytest
import itertools


TEST_BATCH_SIZE = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

inceptions = ['inceptionv3', ]
squeezenets = ['squeezenet1_0', 'squeezenet1_1']
resnets = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
densenets = ['densenet161', 'densenet201', 'densenet169', 'densenet121']
vggs = ['vgg19', 'vgg19_bn', 'vgg16_bn', 'vgg16', 'vgg13_bn', 'vgg13', 'vgg11_bn', 'vgg11']

names = resnets + densenets + squeezenets + vggs + inceptions
pretrained_list = ['imagenet']


@pytest.mark.parametrize(
    'name, pretrained',
    itertools.product(names, pretrained_list)
)
def test_model_forward(name, pretrained, input):
    model_func = getattr(models, name)
    model = model_func(pretrained=pretrained).to(device)
    out = model(input)
    assert tuple(out.shape) == (TEST_BATCH_SIZE, 1000)


@pytest.mark.parametrize(
    'name, pretrained',
    itertools.product(names, pretrained_list)
)
def test_model_features(name, pretrained, input):
    model_func = getattr(models, name)
    model = model_func(pretrained=pretrained).to(device)
    out = model.features(input)
    print(f'out: {out.shape}')


@pytest.fixture
def input():
    return torch.randn(TEST_BATCH_SIZE, 3, 224, 224).to(device)
