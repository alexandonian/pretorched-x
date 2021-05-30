import io
import math
import threading

import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import truncnorm

import torchvision.utils as vutils


def visualize_data(data, num_samples=64, figsize=(15, 15), title='Real Images'):
    if isinstance(data, torch.utils.data.Dataset):
        print(data)
        samples = torch.stack([data[i][0] for i in range(num_samples)])
    elif isinstance(data, torch.utils.data.DataLoader):
        print(data.dataset)
        samples = next(iter(data))[0][:num_samples]
    else:
        raise ValueError(f'Unrecognized data source type: {type(data)}'
                         'Must be instance of either torch Dataset or DataLoader')
    visualize_samples(samples, figsize=figsize, title=title)


def visualize_samples(samples, figsize=(15, 15), title='Samples',
                      nrow=8, padding=5, normalize=True, scale_each=False, use_plt=False, pad_value=0.0):
    # Plot the real images
    im = vutils.make_grid(samples, nrow=nrow, padding=padding,
                          normalize=normalize, scale_each=scale_each, pad_value=pad_value).cpu()
    if use_plt:
        plt.figure(figsize=figsize)
        plt.axis("off")
        plt.title(title)
        plt.imshow(np.transpose(im, (1, 2, 0)))
    else:
        imshow(np.transpose(255 * im, (1, 2, 0)))


def imshow(image, format='png', jpeg_fallback=True):
    image = np.asarray(image, dtype=np.uint8)
    str_file = io.BytesIO()
    Image.fromarray(image).save(str_file, format)
    im_data = str_file.getvalue()
    try:
        disp = IPython.display.display(IPython.display.Image(im_data))
    except IOError:
        if jpeg_fallback and format != 'jpeg':
            print('Warning: image was too large to display in format "{}"; '
                  'trying jpeg instead.').format(format)
            return imshow(image, format='jpeg')
        else:
            raise
    return disp


def smooth_data(data, amount=1.0):
    if not amount > 0.0:
        return data
    data_len = len(data)
    ksize = int(amount * (data_len // 2))
    kernel = np.ones(ksize) / ksize
    return np.convolve(data, kernel, mode='same')


def _save_sample(G, fixed_noise, filename, nrow=8, padding=2, normalize=True):
    fake_image = G(fixed_noise).detach()
    vutils.save_image(fake_image, filename, nrow=nrow, padding=padding, normalize=normalize)


def save_samples(G, fixed_noise, filename, threaded=True):
    if threaded:
        G.to('cpu')
        thread = threading.Thread(name='save_samples',
                                  target=_save_sample,
                                  args=(G, fixed_noise, filename))
        thread.start()
    else:
        _save_sample(G, fixed_noise, filename)


def slerp(start, end, weight):
    """TODO: Finish."""
    low_norm = start / torch.norm(start, dim=1, keepdim=True)
    high_norm = end / torch.norm(end, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(-1))
    so = torch.sin(omega)
    print('ip', (low_norm * high_norm).sum(-1).shape)
    print(f'low_norm: {low_norm.shape}')
    print(f'high_norm: {high_norm.shape}')
    print(f'omega: {omega.shape}')
    print(f'so: {so.shape}')
    # print(f'weight: {weight.shape}')
    print((weight * omega / so).shape)
    # res = ((torch.sin((1.0 - weight) * omega) / so).unsqueeze(1) * start
    #    + (torch.sin(weight * omega) / so).unsqueeze(1) * end)
    res = ((torch.sin((1.0 - weight) * omega) / so) * start
           + (torch.sin(weight * omega) / so) * end)
    return res


def interp(x0, x1, num_midpoints, device='cuda', interp_func=torch.lerp):
    """Interpolate between x0 and x1.

    Args:
        x0 (array-like): Starting coord with shape [batch_size, ...]
        x1 (array-like): Ending coord with shape [batch_size, ...]
        num_midpoints (int): Number of midpoints to interpolate.
        device (str, optional): Device to create interp. Defaults to 'cuda'.
    """
    x0 = x0.view(x0.size(0), 1, *x0.shape[1:])
    x1 = x1.view(x1.size(0), 1, *x1.shape[1:])
    lerp = torch.linspace(0, 1.0, num_midpoints + 2, device=device).to(x0.dtype)
    lerp = lerp.view(1, -1, 1)
    return interp_func(x0, x1, lerp)


def truncated_z_sample(batch_size, dim_z, truncation=1.0, seed=None, device='cuda'):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
    return torch.Tensor(float(truncation) * values).to(device)


def make_grid(tensor, nrow=8):
    """Make a grid of images."""
    tensor = np.array(tensor)
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1]), int(tensor.shape[2])
    grid = np.zeros((height * ymaps, width * xmaps, 3), dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[y * height: (y + 1) * height,
                 x * width: (x + 1) * width] = tensor[k]
            k = k + 1
    return grid

# z0 = torch.randn(2, 12)
# z1 = torch.randn(2, 12)
# z_out = slerp(z0, z1, 0.5)
# # z_out = interp(z0, z1, 4, device='cpu')
# # z_out = interp(z0, z1, 4, device='cpu', interp_func=slerp)

# x = torch.ones(2, 10) * 0
# y = torch.ones(2, 10) * 1
# x = x.view(x.size(0), 1, *x.shape[1:])
# y = y.view(y.size(0), 1, *y.shape[1:])
# w = torch.Tensor([[0.5, 0.8]]).unsqueeze(-1)
# print('x', x.shape)
# print('y', y.shape)
# print('w', w.shape)
# # w = 0.5
# # print(torch.lerp(x, y, w).shape)
# print(slerp(x, y, w).shape)
