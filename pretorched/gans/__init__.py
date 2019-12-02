from . import dcgan
from . import sagan
from . import sngan
from . import utils
from .biggan import BigGAN
from .stylegan import stylegan
from .proggan import proggan

__all__ = [
    'BigGAN',
    'dcgan',
    'sngan',
    'sagan',
    'proggan',
    'stylegan',
    'utils',
]
