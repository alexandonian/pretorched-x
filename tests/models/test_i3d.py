import itertools

import pytest
import torch

from pretorched import models

TEST_BATCH_SIZE = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

name_list = ['i3d', 'i3d_flow']
pretrained_list = [None, 'kinetics-400', 'charades']
num_classes = {'kinetics-400': 400, 'charades': 157}


@pytest.fixture
def input():
    return torch.randn(TEST_BATCH_SIZE, 3, 32, 224, 224).to(device)


@pytest.mark.parametrize(
    'name, pretrained',
    itertools.product(name_list, pretrained_list)
)
def test_i3d(name, pretrained, input):
    model_func = getattr(models, name)
    nc = num_classes.get(pretrained, 400)
    model = model_func(num_classes=nc, pretrained=pretrained).to(device)
    if name in ['i3d_flow']:
        input = input[:, :2]
    out = model(input)
    assert tuple(out.shape) == (TEST_BATCH_SIZE, nc)


@pytest.mark.parametrize(
    'name, pretrained',
    itertools.product(name_list, pretrained_list)
)
def test_i3d_features(name, pretrained, input):
    model_func = getattr(models, name)
    nc = num_classes.get(pretrained, 400)
    model = model_func(num_classes=nc, pretrained=pretrained).to(device)
    if name in ['i3d_flow']:
        input = input[:, :2]
    out = model.features(input)
    assert tuple(out.shape) == (TEST_BATCH_SIZE, 1024)


@pytest.mark.parametrize(
    'name, pretrained',
    itertools.product(name_list, pretrained_list)
)
def test_i3d_forward_features(name, pretrained, input):
    model_func = getattr(models, name)
    nc = num_classes.get(pretrained, 400)
    model = model_func(num_classes=nc, pretrained=pretrained).to(device)
    model.forward = model.features
    if name in ['i3d_flow']:
        input = input[:, :2]
    out = model(input)
    assert tuple(out.shape) == (TEST_BATCH_SIZE, 1024)
