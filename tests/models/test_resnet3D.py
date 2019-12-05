import itertools

import pytest
import torch

from pretorched import models

TEST_BATCH_SIZE = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

name_list = ['resnet3d18', 'resnet3d34', 'resnet3d50']
pretrained_list = [None, 'kinetics-400']


@pytest.mark.parametrize(
    'name, pretrained',
    itertools.product(name_list, pretrained_list)
)
def test_resnets(name, pretrained, input):
    model_func = getattr(models, name)
    model = model_func(pretrained=pretrained).to(device)
    out = model(input)
    assert tuple(out.shape) == (TEST_BATCH_SIZE, 400)


@pytest.fixture
def input():
    return torch.randn(TEST_BATCH_SIZE, 3, 16, 224, 224).to(device)


def test_resnet3d50(input):
    print(f'Using device: {device}')
    model = models.resnet3d50(pretrained=None).to(device)
    out = model(input)
    assert tuple(out.shape) == (TEST_BATCH_SIZE, 400)
