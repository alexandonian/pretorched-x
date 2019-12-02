import functools
from . import biggan


Generator = functools.partial(biggan.Generator, G_attn='0', hier=False, shared_dim=False)

Discriminator = functools.partial(biggan.Discriminator, D_attn='0', D_wide=False)
