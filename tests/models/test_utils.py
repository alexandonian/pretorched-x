import functools
from pretorched.gans import BigGAN, biggan, utils
import pytest
import torch

from pretorched import models
from pretorched.models import utils as mutils


TEST_BATCH_SIZE = 256

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def input():
    return torch.randn(TEST_BATCH_SIZE, 3, 224, 224).to(device)
    # return torch.randn(TEST_BATCH_SIZE, 3, 16, 224, 224).to(device)


def test_elastic_forward(input):
    # model = models.resnet3d50(pretrained=None).to(device)
    model = models.resnet101(pretrained=None).to(device)
    # with torch.no_grad():
    out = mutils.elastic_forward(model, input)
    # assert tuple(out.shape) == (TEST_BATCH_SIZE, 400)
    assert tuple(out.shape) == (TEST_BATCH_SIZE, 1000)


def elastic_gan(model, *input):
    error_msg = 'CUDA out of memory.'

    def chunked_forward(f, *x, chunk_size=1):
        out = []

        for xcs in zip(*[xc.chunk(chunk_size) for xc in x]):
            o = f(*xcs).detach()
            out.append(o)
        return torch.cat(out)

    cs, fit = 1, False
    while not fit:
        try:
            return chunked_forward(model, *input, chunk_size=cs)
        except RuntimeError as e:
            if error_msg in str(e):
                torch.cuda.empty_cache()
                # cs += 1
                cs *= 2
            else:
                raise e


n_classes = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.mark.skipif(device != torch.device('cuda'), reason="Need Cuda to run quickly")
@pytest.mark.parametrize('res', [128, 256, 512])
def test_biggan_generator(res):
    G = biggan.Generator(resolution=res).to(device)
    z_, y_ = utils.prepare_z_y(TEST_BATCH_SIZE, G.dim_z, n_classes, device=device)
    g = functools.partial(G, embed=True)
    G_z = elastic_gan(g, z_, y_)
    assert G_z.size() == torch.Size((TEST_BATCH_SIZE, 3, res, res))
