import pytest

import torch

from pretorched.gans import biggan_deep, BigGANDeep, utils

batch_size = 4
n_classes = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.mark.skipif(device != torch.device('cuda'), reason="Need Cuda to run quickly")
@pytest.mark.parametrize('res', [128, 256, 512])
def test_biggan_deep_generator(res):
    G = biggan_deep.Generator(resolution=res, hier=True).to(device)
    z_, y_ = utils.prepare_z_y(batch_size, G.dim_z, n_classes, device=device)
    G_z = G(z_, G.shared(y_))
    assert G_z.size() == torch.Size((batch_size, 3, res, res))


@pytest.mark.skipif(device != torch.device('cuda'), reason="Need Cuda to run quickly")
@pytest.mark.parametrize('res, pretrained, load_ema', [
    (256, 'places365-challenge', False),
    (256, 'places365-challenge', True),
])
def test_pretrained_biggan_deep(res, pretrained, load_ema):
    G = BigGANDeep(resolution=res, pretrained=pretrained,
                   load_ema=load_ema).to(device)
    z_, y_ = utils.prepare_z_y(batch_size, G.dim_z, n_classes, device=device)
    G_z = G(z_, G.shared(y_))
    assert G_z.size() == torch.Size((batch_size, 3, res, res))
