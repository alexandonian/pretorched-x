import itertools

import pytest
import torch

from pretorched import models

TEST_BATCH_SIZE = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

name_list = ['slowfast18', 'slowfast50', 'slowfast101', 'slowfast152', 'slowfast200']
pretrained_list = [None]
num_classes = {'kinetics-400': 400}


@pytest.fixture
def input():
    return torch.randn(TEST_BATCH_SIZE, 3, 32, 224, 224).to(device)


@pytest.mark.parametrize(
    'name, pretrained',
    itertools.product(name_list, pretrained_list)
)
def test_slowfasts(name, pretrained, input):
    model_func = getattr(models, name)
    nc = num_classes.get(pretrained, 400)
    model = model_func(pretrained=pretrained).to(device)
    out = model(input)
    assert tuple(out.shape) == (TEST_BATCH_SIZE, nc)
