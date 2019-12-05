import itertools

import pytest
import torch

from pretorched import models

TEST_BATCH_SIZE = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

name_list = ['mxresnet18', 'mxresnet34', 'mxresnet50', 'mxresnet101', 'mxresnet152']
pretrained_list = [None]


@pytest.mark.parametrize(
    'name, pretrained',
    itertools.product(name_list, pretrained_list)
)
def test_resnets(name, pretrained, input):
    model_func = getattr(models, name)
    model = model_func(pretrained=pretrained).to(device)
    out = model(input)
    assert tuple(out.shape) == (TEST_BATCH_SIZE, 1000)


@pytest.fixture
def input():
    return torch.randn(TEST_BATCH_SIZE, 3, 224, 224).to(device)

