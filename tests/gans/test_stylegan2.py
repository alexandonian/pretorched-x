from collections import defaultdict
from typing import Dict

import pytest
import torch
from pretorched.gans import stylegan2
from pretorched.gans.stylegan2 import z_sample_for_model

dim_z = 512
batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sizes: Dict[str, int] = defaultdict(lambda: 256, faces=1024, car=512)


@pytest.mark.skipif(device != torch.device('cuda'), reason="Need Cuda to run quickly")
@pytest.mark.parametrize(
    'pretrained',
    [
        'bedroom',
        'car',
        'cat',
        'church',
        'faces',
        'horse',
        'kitchen',
        'places',
    ],
)
def test_pretrained_stylegan2(pretrained):
    G = stylegan2(pretrained=pretrained).to(device)
    z = z_sample_for_model(G, size=batch_size).to(device)
    G_z = G(z)
    assert G_z.size() == torch.Size((batch_size, 3, sizes[pretrained], sizes[pretrained]))
