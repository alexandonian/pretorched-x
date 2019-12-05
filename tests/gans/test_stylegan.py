import pytest

import torch

from pretorched.gans import stylegan

dim_z = 512
batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.mark.skipif(device != torch.device('cuda'), reason="Need Cuda to run quickly")
@pytest.mark.parametrize('pretrained, res, expected', [
    ('ff_hq', 1024, 1024),
    ('celeba_hq', 1024, 1024),
    ('lsun_bedroom', 256, 256),
    ('lsun_car', 512, 512),
    ('lsun_cat', 256, 256),
    ('ff_hq', None, 1024),
    ('celeba_hq', None, 1024),
    ('lsun_bedroom', None, 256),
    ('lsun_car', None, 512),
    ('lsun_cat', None, 256),
])
def test_pretrained_stylegan(pretrained, res, expected):
    G = stylegan(pretrained=pretrained, resolution=res).to(device)
    z = torch.randn(batch_size, dim_z, device=device)
    G_z = G(z)
    assert G_z.size() == torch.Size((batch_size, 3, expected, expected))
