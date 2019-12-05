import pytest

import torch

from pretorched.gans import proggan

dim_z = 512
batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.mark.skipif(device != torch.device('cuda'), reason="Need Cuda to run quickly")
@pytest.mark.parametrize('pretrained, res, expected', [
    ('lsun_bedroom', 256, 256),
    ('lsun_bedroom', None, 256),
    ('lsun_church', 256, 256),
    ('lsun_church', None, 256),
    ('lsun_diningroom', 256, 256),
    ('lsun_diningroom', None, 256),
    ('lsun_kitchen', 256, 256),
    ('lsun_kitchen', None, 256),
    ('lsun_livingroom', 256, 256),
    ('lsun_livingroom', None, 256),
    ('lsun_restaurant', 256, 256),
    ('lsun_restaurant', None, 256),
    ('celeba_hq', 1024, 1024),
    ('celeba_hq', None, 1024),
])
def test_pretrained_proggan(pretrained, res, expected):
    G = proggan(pretrained=pretrained, resolution=res).to(device)
    z = torch.randn(batch_size, dim_z, device=device)
    G_z = G(z)
    assert G_z.size() == torch.Size((batch_size, 3, expected, expected))
