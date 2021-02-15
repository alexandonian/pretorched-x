####
# This port of styleganv2 is derived from and perfectly compatible with
# the pytorch port by https://github.com/rosinality/stylegan2-pytorch.
#
# In this reimplementation, all non-leaf modules are subclasses of
# nn.Sequential so that the network can be more easily split apart
# for surgery and direct rewriting.

import os
from collections import defaultdict
from typing import Dict
from pretorched.gans.stylegan import RESOLUTIONS

import numpy
import torch
from torch.utils import model_zoo
from torch.utils.data import TensorDataset

from .models import SeqStyleGAN2

# TODO: change these paths to non-antonio paths, probably load from url if not exists
WEIGHT_URLS = 'http://wednesday.csail.mit.edu/placesgan/tracer/utils/stylegan2/weights/'
root_url = 'http://pretorched-x.csail.mit.edu/gans/StyleGAN2'
sizes: Dict[str, int] = defaultdict(lambda: 256, faces=1024, car=512)

model_urls = {
    'bedroom': {256: {'G': os.path.join(root_url, 'bedroom_256x256_G-11072c2b.pth')}},
    'car': {512: {'G': os.path.join(root_url, 'car_512x512_G-0ce43c2d.pth')}},
    'cat': {256: {'G': os.path.join(root_url, 'cat_256x256_G-f0b1bf2c.pth')}},
    'church': {256: {'G': os.path.join(root_url, 'church_256x256_G-905a491c.pth')}},
    'faces': {1024: {'G': os.path.join(root_url, 'faces_1024x1024_G-c10dbc80.pth')}},
    'horse': {256: {'G': os.path.join(root_url, 'horse_256x256_G-524e68e4.pth')}},
    'kitchen': {256: {'G': os.path.join(root_url, 'kitchen_256x256_G-d7083483.pth')}},
    'places': {256: {'G': os.path.join(root_url, 'places_256x256_G-47082c7d.pth')}},
}


def load_state_dict(category):
    chkpt_name = f'stylegan2_{category}.pt'
    model_path = os.path.join('weights', chkpt_name)
    os.makedirs('weights', exist_ok=True)

    if not os.path.exists(model_path):
        url = WEIGHT_URLS + chkpt_name
        state_dict = model_zoo.load_url(url, model_dir='weights', progress=True)
        torch.save(state_dict, model_path)
    else:
        state_dict = torch.load(model_path)
    return state_dict


def load_seq_stylegan(category, truncation=1.0, **kwargs):  # mconv='seq'):
    ''' loads nn sequential version of stylegan2 and puts on gpu'''
    state_dict = load_state_dict(category)
    size = sizes[category]
    g = SeqStyleGAN2(size, style_dim=512, n_mlp=8, truncation=truncation, **kwargs)
    g.load_state_dict(state_dict['g_ema'], latent_avg=state_dict['latent_avg'])
    g.cuda()
    return g


def stylegan2(
    pretrained='bedroom', truncation=1.0, resolution=None, **kwargs
):  # mconv='seq'):
    ''' loads nn sequential version of stylegan2.'''
    if pretrained is not None:
        resolution = sizes.get(pretrained, 256) if resolution is None else resolution
        url = model_urls[pretrained][resolution]['G']
        state_dict = torch.hub.load_state_dict_from_url(url)
        g = SeqStyleGAN2(
            resolution, style_dim=512, n_mlp=8, truncation=truncation, **kwargs
        )
        g.load_state_dict(state_dict['g_ema'], latent_avg=state_dict['latent_avg'])
    else:
        assert resolution is not None, 'Must specify pretrained model or resolution!'
        g = SeqStyleGAN2(
            resolution, style_dim=512, n_mlp=8, truncation=truncation, **kwargs
        )
    return g


def z_dataset_for_model(model, size=100, seed=1, indices=None):
    if indices is not None:
        indices = torch.as_tensor(indices, dtype=torch.int64, device='cpu')
        zs = z_sample_for_model(model, indices.max().item() + 1, seed)
        zs = zs[indices]
    else:
        zs = z_sample_for_model(model, size, seed)
    return TensorDataset(zs)


def z_sample_for_model(model, size=100, seed=1):
    # If the model is marked with an input shape, use it.
    if hasattr(model, 'input_shape'):
        sample = standard_z_sample(size, model.input_shape[1], seed=seed).view(
            (size,) + model.input_shape[1:]
        )
        return sample
    # Examine first conv in model to determine input feature size.
    first_layer = [
        c
        for c in model.modules()
        if isinstance(c, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear))
    ][0]
    # 4d input if convolutional, 2d input if first layer is linear.
    if isinstance(first_layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        sample = standard_z_sample(size, first_layer.in_channels, seed=seed)[
            :, :, None, None
        ]
    else:
        sample = standard_z_sample(size, first_layer.in_features, seed=seed)
    return sample


def standard_z_sample(size, depth, seed=1, device=None):
    '''
    Generate a standard set of random Z as a (size, z_dimension) tensor.
    With the same random seed, it always returns the same z (e.g.,
    the first one is always the same regardless of the size.)
    '''
    # Use numpy RandomState since it can be done deterministically
    # without affecting global state
    rng = numpy.random.RandomState(seed)
    result = torch.from_numpy(
        rng.standard_normal(size * depth).reshape(size, depth)
    ).float()
    if device is not None:
        result = result.to(device)
    return result
