"""StyleGAN.

This module implements teh Generative Adversarial Network described in:

A Style-Based Generator Architecture for Generative Adversarial Networks
Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA)
http://stylegan.xyz/paper

Code derived from:
https://github.com/SsnL/stylegan
"""
import collections
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIM_Z = collections.defaultdict(lambda: 512)

FULL_RESOLUTIONS = {
    'lsun_car': (512, 384),
    'ff_hq': (1024, 1024),
    'celeba_hq': (1024, 1024),
    'lsun_bedroom': (256, 256),
    'lsun_cat': (256, 256),
}
RESOLUTIONS = {
    'ff_hq': 1024,
    'celeba_hq': 1024,
    'lsun_bedroom': 256,
    'lsun_car': 512,
    'lsun_cat': 256,
}

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
root_url = 'http://pretorched-x.csail.mit.edu/gans/StyleGAN'

model_urls = {
    'celeba_hq': {
        1024: {
            'G': os.path.join(root_url, 'celeba_hq_1024x1024_G-c8acef81.pth'),
        },
    },
    'ff_hq': {
        1024: {
            'G': os.path.join(root_url, 'ff_hq_1024x1024_G-21a7044d.pth'),
        }
    },
    'lsun_bedroom': {
        256: {
            'G': os.path.join(root_url, 'lsun_bedroom_256x256_G-da907d98.pth'),
        },
    },
    'lsun_car': {
        512: {
            'G': os.path.join(root_url, 'lsun_car_512x384_G-d2188b0a.pth'),
        },
    },
    'lsun_cat': {
        256: {
            'G': os.path.join(root_url, 'lsun_cat_256x256_G-384e9e73.pth'),
        },
    },
}


def stylegan(pretrained='ff_hq', resolution=None):
    if pretrained is not None:
        resolution = RESOLUTIONS.get(pretrained) if resolution is None else resolution
        url = model_urls[pretrained][resolution]['G']
        state_dict = torch.hub.load_state_dict_from_url(url)
        net = G(out_res=resolution)
        net.load_state_dict(state_dict)
    else:
        assert resolution is not None, 'Must specify pretrained model or resolution!'
        net = G(out_res=max(resolution))
    return net


class NonLinearityMeta(type):
    def __call__(cls, *args, **kwargs):
        return cls.activate(*args, **kwargs)


class NonLinearity(object, metaclass=NonLinearityMeta):
    gain = NotImplemented
    activate = NotImplemented


class ReLU(NonLinearity):
    gain = np.sqrt(2)
    activate = F.relu


class LeakyReLU(NonLinearity):
    gain = np.sqrt(2)

    @staticmethod
    def activate(x, inplace=False):
        return F.leaky_relu(x, negative_slope=0.2, inplace=inplace)


class ScaledParamModule(nn.Module):
    # linear w: [ fan_out, fan_in ]
    # conv   w: [  nc_out,  nc_in, k1, k2 ]
    # convT  w: [  nc_in,  nc_out, k1, k2 ], but let's ignore this case because
    #    (1) the tf impl doesn't special-case
    #    (2) convT is only used for fusing Upsample & Conv2d, and in that case, the
    #        weight should be done as if it is for a Conv2d.
    #
    # NB: in tf code, use_wscale has default value False, but for StyleGAN it is
    #     True everywhere, so I changed it.
    def scale_weight(self, gain=np.sqrt(2), use_wscale=True, lrmul=1, new_name='_weight'):
        weight = self.weight
        assert isinstance(weight, nn.Parameter)

        fan_in = np.prod(weight.shape[1:])
        he_std = gain / np.sqrt(fan_in)  # He init

        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            runtime_coef = he_std * lrmul
        else:
            init_std = he_std / lrmul
            runtime_coef = lrmul

        # Init variable using He init.
        weight.data.normal_(0, init_std)

        # add scale hook
        self.add_scale_hook('weight', new_name, runtime_coef)

    def scale_bias(self, lrmul=1, new_name='_bias'):
        if self.bias is None:
            assert not hasattr(self, new_name)
            # do not delete so we don't have to restore in forward
            # del self.bias
            self.register_parameter(new_name, None)
            return

        bias = self.bias
        assert isinstance(bias, nn.Parameter)

        # zero out
        bias.data.zero_()

        # add scale hook
        self.add_scale_hook('bias', new_name, lrmul)

    def add_scale_hook(self, name, new_name, coef):
        param = getattr(self, name)
        assert isinstance(param, nn.Parameter)
        assert not hasattr(self, new_name)

        delattr(self, name)
        self.register_parameter(new_name, param)
        # Note that the following line uses `m` rather than `self`, and thus
        # doesn't maintaing the reference and allows for deep copying.
        self.register_forward_pre_hook(lambda m, inp: setattr(m, name, getattr(m, new_name) * coef))


class ScaledParamLinear(nn.Linear, ScaledParamModule):
    def __init__(self, *args, gain=np.sqrt(2), use_wscale=True, lrmul=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_weight(gain, use_wscale, lrmul)
        self.scale_bias(lrmul)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self._bias is not None  # use the _real param
        )


class ScaledParamConv2d(nn.Conv2d, ScaledParamModule):
    def __init__(self, *args, gain=np.sqrt(2), use_wscale=True, lrmul=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_weight(gain, use_wscale, lrmul)
        self.scale_bias(lrmul)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self._bias is None:  # use the _real param
            s += ', bias=False'
        return s.format(**self.__dict__)


class UpConv2d(ScaledParamConv2d):
    # Fuse Upsample 2x and Conv2d if desirable.

    # NOTE [ Fusing Nearest Neighbor Upsampling and Conv2d to a ConvTranspose2d ]
    #
    # For exact match, we should flip the kernel along the spatial
    # dimensions, e.g., with a `.flip(2, 3)`.
    # This is because we will calculate the sum combinations in kernel
    # and then apply convT with stride so that each input pixel hits the
    # exact same kernel values as it would with upsample + conv, but
    # now summed as a single value. In ConvT, kernel and input are in
    # reversed space, in the sense that the top-left input pixel sees
    # top-left kernel region in conv, but bottom-right in convT.
    # However, the tf code doesn't do this and this is also a problem in
    # tf, so to keep trained weights compatibility, we don't flip.

    def __init__(self, in_res, in_channels, out_channels, kernel_size, padding=0,
                 bias=True, gain=np.sqrt(2), use_wscale=True, lrmul=1):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias,
                         gain=gain, use_wscale=use_wscale, lrmul=lrmul)
        self.do_fuse = in_res * 2 >= 128

    def forward(self, x):
        if self.do_fuse:
            w = F.pad(self.weight.transpose(0, 1), (1, 1, 1, 1))
            w = w[:, :, 1:, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, 1:] + w[:, :, :-1, :-1]
            return F.conv_transpose2d(x, w, self.bias, stride=(2, 2), padding=self.padding)
        else:
            return nn.Conv2d.forward(self, F.interpolate(x, scale_factor=2))


class Blur2d(nn.Module):
    def __init__(self, kernel=[1, 2, 1], padding=1, normalize=True, flip=False, stride=1):
        super().__init__()

        self.stride = stride
        self.padding = padding

        # build kernel
        kernel = torch.as_tensor(kernel, dtype=torch.get_default_dtype())
        if kernel.dim() == 1:
            kernel = kernel[:, None] * kernel
        assert kernel.dim() == 2

        if normalize:
            kernel /= kernel.sum()

        if flip:
            kernel = kernel.flip(0, 1)

        if kernel.numel() == 1 and kernel[0, 0].item() == 1:
            # No-op => early exit.
            self.no_conv = True
        else:
            self.no_conv = False
            # prepare for conv2d
            # use single channel (merge nc to batch dimension)
            self.register_buffer('kernel', kernel.expand(1, 1, -1, -1).contiguous())

    def forward(self, x):
        if self.no_conv:
            if self.stride == 1:
                return x
            else:
                return x[:, :, ::self.stride, ::self.stride]
        else:
            b, nc, h, w = x.size()
            y = F.conv2d(x.view(-1, 1, h, w), self.kernel, bias=None, stride=self.stride, padding=self.padding)
            return y.view(b, nc, y.size(2), y.size(3))


class MappingG(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, dim_latent=512, n_layers=8,
                 nonlinearity=LeakyReLU, use_wscale=True,
                 use_class_labels=False, nlabels=1, embed_size=0, lrmul=0.01):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.dim_latent = dim_latent
        assert n_layers >= 1
        self.n_layers = n_layers
        self.act = nonlinearity
        scale_param_opt = dict(gain=self.act.gain, lrmul=lrmul, use_wscale=use_wscale)

        self.use_class_labels = use_class_labels

        if self.use_class_labels:
            self.embedding = nn.Embedding(nlabels, embed_size)

        dim = z_dim + embed_size if use_class_labels else z_dim

        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            self.fcs.append(
                ScaledParamLinear(dim, dim_latent if i < (n_layers - 1) else w_dim, **scale_param_opt),
            )
            dim = dim_latent

    def forward(self, z, y=None):
        if self.use_class_labels:
            yembed = self.embedding(y)
            yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
            z = torch.cat([z, yembed], dim=1)
        # NB: this is not z.norm(p=2, dim=1, keepdim=True)!!!!
        z = z * z.pow(2).mean(dim=1, keepdim=True).add_(1e-8).rsqrt()
        for fc in self.fcs:
            z = self.act(fc(z))
        return z

    def __repr__(self):
        return '{}(z_dim={}, w_dim={}, dim_latent={}, n_layers={}, ...)'.format(
            self.__class__.__name__, self.z_dim, self.w_dim, self.dim_latent, self.n_layers)


class SynthesisG(nn.Module):
    class AddNoise(ScaledParamModule):
        # `B` block + Noise in Fig1
        def __init__(self, nc, res):
            super().__init__()
            self.res = res
            self.weight = nn.Parameter(torch.zeros(nc, 1, 1, requires_grad=True))
            self.bias = nn.Parameter(torch.zeros(nc, 1, 1, requires_grad=True))
            self.scale_bias(lrmul=1)

        def forward(self, x, noise=None):
            if noise is None:
                noise = torch.randn(x.size(0), 1, self.res, self.res, device=x.device, dtype=x.dtype)
            return x + noise * self.weight + self.bias

    class AffineStyle(nn.Module):
        # `A` block + AdaIN in Fig1
        def __init__(self, w_dim, nc):
            super().__init__()
            self.nc = nc
            self.fc = ScaledParamLinear(w_dim, nc * 2, gain=1)

        def forward(self, x, w):
            normalized = F.instance_norm(x, weight=None, bias=None, eps=1e-8)
            affine_params = self.fc(w).view(-1, 2, self.nc, 1, 1)
            return normalized * (affine_params[:, 0].add_(1)) + affine_params[:, 1]

    class Block(nn.Module):
        def __init__(self, w_dim, in_res, in_nc, out_res, out_nc,
                     blur_filter=[1, 2, 1],
                     nonlinearity=LeakyReLU, use_wscale=True, lrmul=1,
                     skip_first_layer=False):  # skip_first_layer skips the upsample & first conv, used for first block
            super().__init__()
            self.skip_first_layer = skip_first_layer
            self.act = nonlinearity

            scale_param_opt = dict(gain=self.act.gain, lrmul=lrmul, use_wscale=use_wscale)

            # NB: the following (up)conv* layers have bias=False, because we
            #     assume that we are always using noise, and the bias is applied
            #     in noise* layers.  This is still consistent with official tf
            #     code.
            if not self.skip_first_layer:
                self.upconv1 = UpConv2d(in_res, in_nc, out_nc, 3, padding=1, bias=False, **scale_param_opt)

                assert len(blur_filter) % 2 == 1
                self.blur1 = Blur2d(blur_filter, padding=(len(blur_filter) >> 1))

            self.noise1 = SynthesisG.AddNoise(out_nc, out_res)
            self.style1 = SynthesisG.AffineStyle(w_dim, out_nc)

            self.conv2 = ScaledParamConv2d(out_nc, out_nc, 3, padding=1, bias=False, **scale_param_opt)
            self.noise2 = SynthesisG.AddNoise(out_nc, out_res)
            self.style2 = SynthesisG.AffineStyle(w_dim, out_nc)

        def forward(self, x, ws, noises=(None, None)):
            if not self.skip_first_layer:
                x = self.blur1(self.upconv1(x))

            x = self.noise1(x, noises[0])
            x = self.act(x)
            x = self.style1(x, ws[0])

            x = self.conv2(x)
            x = self.noise2(x, noises[1])
            x = self.act(x)
            x = self.style2(x, ws[1])
            return x

    def __init__(self, w_dim=512, image_out_nc=3, image_out_res=1024,
                 nc_base=8192, nc_decay=1.0, nc_max=512,
                 nonlinearity=LeakyReLU, use_wscale=True, lrmul=1):
        super().__init__()
        self.out_res = image_out_res
        log_image_out_res = int(np.log2(image_out_res))
        assert image_out_res == 2 ** log_image_out_res and image_out_res >= 4

        # output nc of a block.
        #
        # log_res refers to the input to the block, which is immediately
        # upsampled.
        #
        # In the first block, there is no upsample, and input is directly 4x4,
        # but you should still treat as if it is upsampled from 2x2 and use
        # log_res=1.
        def get_out_nc(log_res):
            return min(int(nc_base / 2 ** (log_res * nc_decay)), nc_max)

        self.const = nn.Parameter(torch.ones(1, get_out_nc(1), 4, 4, requires_grad=True))

        # start at 4x4
        in_res = 2
        in_nc = None  # first shouldn't matter
        for in_log_res in range(1, log_image_out_res):
            out_res = in_res * 2
            out_nc = get_out_nc(in_log_res)

            b = SynthesisG.Block(
                w_dim, in_res, in_nc, out_res, out_nc,
                skip_first_layer=(in_log_res == 1),
                nonlinearity=nonlinearity, use_wscale=use_wscale, lrmul=lrmul,
            )
            self.add_module('{res}x{res}'.format(res=out_res), b)

            to_rgb = ScaledParamConv2d(out_nc, image_out_nc, 1, gain=1, use_wscale=use_wscale, lrmul=lrmul)
            out_log_res = in_log_res + 1
            self.add_module('{res}x{res}_to_rgb_lod{lod}'.format(
                res=out_res, lod=(log_image_out_res - out_log_res)), to_rgb)

            in_res = out_res
            in_nc = out_nc

        assert in_res == image_out_res
        self.num_blocks = len(self.blocks)
        self.num_layers = self.num_blocks * 2

    @property
    def blocks(self):
        blocks = []
        children_dict = {}
        for name, module in self.named_children():
            children_dict[name] = module

        log_out_res = int(np.log2(self.out_res))
        out_res = 4
        for _ in range(1, log_out_res):
            name = '{res}x{res}'.format(res=out_res)
            module = children_dict[name]
            blocks.append(module)
            out_res = out_res * 2

        return blocks

    @property
    def rgb_convs(self):
        rgb_convs = []
        children_dict = {}
        for name, module in self.named_children():
            children_dict[name] = module

        log_out_res = int(np.log2(self.out_res))
        out_res = 4
        for in_log_res in range(1, log_out_res):
            out_log_res = in_log_res + 1
            name = '{res}x{res}_to_rgb_lod{lod}'.format(res=out_res, lod=(log_out_res - out_log_res))
            module = children_dict[name]
            rgb_convs.append(module)
            out_res = out_res * 2

        return rgb_convs

    # allow taking in a list of W for style mixing
    def forward(self, ws, lod=0, alpha=1, noises=None):
        blocks = self.blocks
        rgb_convs = self.rgb_convs

        assert 0 <= lod < len(blocks)
        stop_after = len(blocks) - lod - 1

        num_layers = (stop_after + 1) * 2

        if isinstance(ws, torch.Tensor) and ws.dim() == 3:
            ws = ws.unbind(dim=1)  # assuming its [batch x num_layer x w]
        if not isinstance(ws, collections.abc.Sequence):
            ws = [ws for _ in range(num_layers)]

        if not isinstance(noises, collections.abc.Sequence):
            noises = [noises for _ in range(num_layers)]

        x = self.const.expand(ws[0].size(0), -1, -1, -1)

        for i, b in enumerate(blocks):
            block_extra_inp_indices = slice(i * 2, i * 2 + 2)
            x = b(x, ws[block_extra_inp_indices], noises=noises[block_extra_inp_indices])
            if i == stop_after - 1:
                y = F.interpolate(x, scale_factor=2)
                y = rgb_convs[stop_after - 1](y)
            if i == stop_after:
                x = rgb_convs[i](x)
                return x if stop_after == 0 else (1 - alpha) * y + alpha * x


class G(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, out_nc=3, out_res=1024, use_class_labels=False,
                 w_avg_beta=0.995,           # find moving average of w,                    in training
                 style_mixing_prob=0.995,    # prob of applying style mixing,               in training
                 truncation_psi=0.7,         # mixing rate to w_avg for truncation,         in eval
                 truncation_cutoff=8,        # layer cutoff index index for truncation,     in eval
                 nlabels=1,                  # number of classes (if using class labels)
                 embed_size=256,             # embedding size for encoding class label information
                 nonlinearity=LeakyReLU, use_wscale=True, **kwargs):
        super().__init__()
        self.register_buffer('w_avg', torch.zeros(w_dim))
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.out_nc = out_nc
        self.out_res = out_res
        self.w_avg_beta = w_avg_beta
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff
        self.mapping = MappingG(z_dim, w_dim, nonlinearity=nonlinearity,
                                use_wscale=use_wscale,
                                use_class_labels=use_class_labels,
                                nlabels=nlabels, embed_size=embed_size)
        self.synthesis = SynthesisG(w_dim, out_nc, out_res, nonlinearity=nonlinearity, use_wscale=use_wscale)

    def forward(self, z, y=None, lod=0, alpha=1, w=None, noises=None, get_w=False,
                w_avg_beta=None, style_mixing_prob=None,
                truncation_psi=None, truncation_cutoff=None):

        # really is the total number of layers for the synthesis network
        total_num_layers = self.synthesis.num_layers
        forward_num_layers = total_num_layers - lod * 2

        if w is None:
            assert z is not None and z.dim() == 2 and z.size(1) == self.z_dim
            w = self.mapping(z, y)
        else:
            assert w.dim() == 2 and w.size(1) == self.w_dim

        if get_w:
            mapping_output_w = w.clone()

        ws = [w for _ in range(total_num_layers)]

        if self.training:
            # update moving average
            if w_avg_beta is None:
                w_avg_beta = self.w_avg_beta
            if w_avg_beta != 1:
                with torch.no_grad():
                    torch.lerp(w.mean(0), self.w_avg, self.w_avg_beta, out=self.w_avg)

            # style mixing
            if style_mixing_prob is None:
                style_mixing_prob = self.style_mixing_prob
            if style_mixing_prob > 0 and torch.rand((), device='cpu').item() < style_mixing_prob:
                w2 = self.mapping(torch.randn_like(z), y)
                cutoff = int(torch.randint(low=1, high=forward_num_layers, size=(), device='cpu').item())
                # w for < cutoff; w2 for >= cutoff
                ws = ws[:cutoff] + [w2 for _ in range(forward_num_layers - cutoff)]

        else:
            # truncation
            if truncation_psi is None:
                truncation_psi = self.truncation_psi
            if truncation_cutoff is None:
                truncation_cutoff = self.truncation_cutoff

            # truncate for < cutoff
            if truncation_cutoff > 0 and truncation_psi != 1:
                expanded_avg_w = self.w_avg.expand_as(ws[0])

                # in eval part, current code implies that ws is a list of the same w
                # tensor repeated many times since there is no style mixing, but
                # let's be general and detect before optimizing for that.
                if all(_w is w for _w in ws):
                    truncate_w = torch.lerp(expanded_avg_w, w, truncation_psi)
                    ws = [truncate_w for _ in range(truncation_cutoff)] + ws[:(forward_num_layers - truncation_cutoff)]
                else:
                    for i in range(truncation_cutoff):
                        # use out-of-place because these ws may be references to the
                        # same tensor
                        ws[i] = torch.lerp(expanded_avg_w, w[i], truncation_psi)

        ims = self.synthesis(ws, lod=lod, noises=noises, alpha=alpha)

        if get_w:
            return mapping_output_w, ims
        else:
            return ims

    def __repr__(self):
        return '{}(z_dim={}, w_dim={}, out_nc={}, out_res={}, ...)'.format(
            self.__class__.__name__, self.z_dim, self.w_dim, self.out_nc, self.out_res)

    def convert_tf_weights_to_state_dict(self, tf_net, device=torch.device('cpu')):
        # full of hacks
        state_dict = {}

        def raise_unexpected(tf_k, v):
            raise RuntimeError("Unexpected key '{}' with shape: {}".format(tf_k, v.size()))

        def transform_conv_w(v):
            # tf: [ k1, k2, nc_in, nc_out ]
            # pt: [ nc_out, nc_in, k1, k2 ]
            return v.permute(3, 2, 0, 1)

        def transform_fc_w(v):
            # tf: [ fan_in, fan_out ]
            # pt: [ fan_out, fan_in ]
            return v.t()

        def sub_synthesis(k, tf_k, v):
            k = k.replace('G_synthesis.', 'synthesis.')

            def replace_tail_if_match(pattern, new):
                nonlocal k
                if k.endswith(pattern):
                    k = k[:-len(pattern)] + new
                    return True
                return False

            # deal with the first block
            if k == 'synthesis.4x4.Const.const':
                return 'synthesis.const', v
            elif k.startswith('synthesis.4x4.'):
                if 'synthesis.4x4.Const' in k:
                    k = k.replace('synthesis.4x4.Const', 'synthesis.4x4.Conv0_up')
                elif 'synthesis.4x4.Conv' in k:
                    k = k.replace('synthesis.4x4.Conv', 'synthesis.4x4.Conv1')
                else:
                    raise_unexpected(tf_k, v)

            if replace_tail_if_match('.Conv0_up.weight', '.upconv1._weight'):  # noqa: E241, E202
                return k, transform_conv_w(v)
            if replace_tail_if_match('.Conv0_up.Noise.weight', '.noise1.weight'):  # noqa: E241, E202
                return k, v.view(-1, 1, 1)
            if replace_tail_if_match('.Conv0_up.bias', '.noise1._bias'):  # noqa: E241, E202
                return k, v.view(-1, 1, 1)
            if replace_tail_if_match('.Conv0_up.StyleMod.weight', '.style1.fc._weight'):  # noqa: E241, E202
                return k, transform_fc_w(v)
            if replace_tail_if_match('.Conv0_up.StyleMod.bias', '.style1.fc._bias'):  # noqa: E241, E202
                return k, v

            if 'Conv0_up' in k:
                raise_unexpected(tf_k, v)

            if replace_tail_if_match('.Conv1.weight', '.conv2._weight'):  # noqa: E241, E202
                return k, transform_conv_w(v)
            if replace_tail_if_match('.Conv1.Noise.weight', '.noise2.weight'):  # noqa: E241, E202
                return k, v.view(-1, 1, 1)
            if replace_tail_if_match('.Conv1.bias', '.noise2._bias'):  # noqa: E241, E202
                return k, v.view(-1, 1, 1)
            if replace_tail_if_match('.Conv1.StyleMod.weight', '.style2.fc._weight'):  # noqa: E241, E202
                return k, transform_fc_w(v)
            if replace_tail_if_match('.Conv1.StyleMod.bias', '.style2.fc._bias'):  # noqa: E241, E202
                return k, v

            if 'Conv1' in k:
                raise_unexpected(tf_k, v)

            m = re.match(r'^synthesis\.ToRGB_lod(\d+)\.(weight|bias)$', k)
            if m:
                lod = int(m.group(1))
                k = 'synthesis.{res}x{res}_to_rgb_lod{lod}._{name}'.format(
                    res=int(self.out_res / 2 ** lod), lod=lod, name=m.group(2))
                if m.group(2) == 'weight':
                    v = transform_conv_w(v)
                return k, v

            raise_unexpected(tf_k, v)

        for tf_k, tf_v in tf_net.vars.items():
            assert '.' not in tf_k

            k = tf_k.replace('/', '.')
            v = torch.as_tensor(tf_v.eval())

            if k in {'lod', 'G_synthesis.lod'} or k.startswith('G_synthesis.noise'):
                # no input buffer
                continue

            elif k == 'dlatent_avg':
                k = 'w_avg'

            elif k.startswith('G_synthesis.'):
                k, v = sub_synthesis(k, tf_k, v)

            elif k.startswith('G_mapping.'):
                m = re.match(r'^G_mapping\.Dense(\d+)\.(weight|bias)$', k)
                if not m:
                    raise_unexpected(tf_k, v)

                k = 'mapping.fcs.{}._{}'.format(m.group(1), m.group(2))
                if m.group(2) == 'weight':
                    v = transform_fc_w(v)

            else:
                raise_unexpected(tf_k, v)

            state_dict[k] = v

        # tf state dict doesn't have the blur kernels, but pytorch wants to see
        # them
        for k, v in self.state_dict().items():
            if re.match(r'^synthesis\.\d+x\d+\.blur1\.kernel$', k):
                state_dict[k] = v.detach()

        # device & contiguity
        for k in state_dict:
            state_dict[k] = state_dict[k].to(device).contiguous()

        return state_dict

    @property
    def dim_z(self):
        return self.z_dim


class D(nn.Module):
    # Discriminator from https://arxiv.org/pdf/1710.10196.pdf
    class MinibatchStddev(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, group_size=4):
            '''
            Implements the MinibatchStddevLayer from https://arxiv.org/pdf/1710.10196.pdf

            group size: int, must divide the batch size
            in: BS x C x H x W
            out: BS x (C + 1) x H x W
            '''
            s = x.size()
            group_size = min(group_size, s[0])
            y = x.view(group_size, -1, s[1], s[2], s[3])
            y = y - torch.mean(y, 0, keepdim=True)
            y = (y**2).mean(0, keepdim=False)
            y = torch.sqrt(y + 10**-8)
            # y = y.mean((1, 2, 3), keepdim=True).expand_as(x)
            y = y.mean((1, 2, 3), keepdim=True).repeat(group_size, 1, s[2], s[3])
            return torch.cat([x, y], 1)

    class ConvBlock(nn.Module):
        def __init__(self, in_nc, out_nc, last_layer=False, nonlinearity=LeakyReLU, use_wscale=True, lrmul=1.0):
            super().__init__()
            self.act = nonlinearity
            self.last_layer = last_layer
            scale_param_opt = dict(gain=self.act.gain, lrmul=lrmul, use_wscale=use_wscale)

            if not self.last_layer:
                self.blur = Blur2d(kernel=[1, 2, 1], normalize=True, stride=1, padding=1)
                self.pool = Blur2d(kernel=[0.5, 0.5], normalize=False, stride=2, padding=0)
                self.conv1 = ScaledParamConv2d(in_nc, in_nc, 3, padding=1, bias=True, **scale_param_opt)
                self.conv2 = ScaledParamConv2d(in_nc, out_nc, 3, padding=1, bias=True, **scale_param_opt)
            else:
                self.minibatch_stddev = D.MinibatchStddev()
                self.conv1 = ScaledParamConv2d(in_nc + 1, in_nc, 3, padding=1, bias=True, **scale_param_opt)
                self.conv2 = ScaledParamLinear(in_nc * 16, out_nc, bias=True, **scale_param_opt)

        def forward(self, x):
            if self.last_layer:
                x = self.minibatch_stddev(x)
            out = self.act(self.conv1(x))
            if self.last_layer:
                out = out.view(out.size(0), -1)
            else:
                out = self.blur(out)
            # out = self.act(self.conv2(out)) #possible change in order
            out = self.conv2(out)
            if not self.last_layer:
                out = self.pool(out)
            return out

    def __init__(self, image_in_nc=3, out_res=1024,
                 nc_base=16, nc_decay=1.0, nc_max=512,
                 nonlinearity=LeakyReLU, use_wscale=True, use_class_labels=False, nlabels=None, lrmul=1, **kwargs):
        super().__init__()
        self.out_res = out_res
        log_out_res = int(np.log2(out_res))
        assert out_res == 2 ** log_out_res and out_res >= 4

        # output nc of a block.
        #
        # log_res refers to the input to the block, which is immediately
        # upsampled.
        #
        # In the first block, there is no upsample, and input is directly 4x4,
        # but you should still treat as if it is upsampled from 2x2 and use
        # log_res=1.
        def get_in_nc(log_res):
            return min(int(nc_base * 2**(log_res - 1)), nc_max)

        def get_out_nc(log_res):
            return min(int(nc_base * 2**log_res), nc_max)

        # plain list
        # we will register them using more meaningful names, mainly to be easier
        # loading tf weights, which are stored in namespaces alike 4x4, 16x16,
        # etc.

        # start at 4x4
        in_res = 2
        # first shouldn't matter
        for in_log_res in reversed(range(1, log_out_res)):
            out_res = in_res * 2
            in_nc = get_in_nc(in_log_res)
            out_nc = get_out_nc(in_log_res)

            from_rgb = ScaledParamConv2d(image_in_nc, in_nc, kernel_size=1, gain=nonlinearity.gain,
                                         bias=True, use_wscale=use_wscale, lrmul=lrmul)

            b = D.ConvBlock(in_nc, out_nc, last_layer=(in_log_res == log_out_res - 1),
                            nonlinearity=nonlinearity, use_wscale=use_wscale, lrmul=lrmul)

            self.add_module('{res}x{res}'.format(res=out_res), b)

            out_log_res = in_log_res + 1
            self.add_module('{res}x{res}_from_rgb_lod{lod}'.format(
                res=out_res, lod=(out_log_res - 2)), from_rgb)

            in_res = out_res

        assert in_res == out_res

        self.num_blocks = len(self.blocks)
        self.num_layers = self.num_blocks * 2

        self.use_class_labels = use_class_labels
        self.avgpool = Blur2d(kernel=[0.5, 0.5], normalize=False, stride=2, padding=0)
        if self.use_class_labels:
            self.fc = ScaledParamLinear(min(get_out_nc(log_out_res - 1), nc_max), nlabels,
                                        gain=1.0, bias=True, use_wscale=True, lrmul=lrmul)
        else:
            self.fc = ScaledParamLinear(min(get_out_nc(log_out_res - 1), nc_max), 1,
                                        gain=1.0, bias=True, use_wscale=True, lrmul=lrmul)

    @property
    def blocks(self):
        blocks = []

        children_dict = {}
        for name, module in self.named_children():
            children_dict[name] = module

        log_out_res = int(np.log2(self.out_res))
        out_res = 4
        for _ in reversed(range(1, log_out_res)):
            name = '{res}x{res}'.format(res=out_res)
            module = children_dict[name]
            blocks.append(module)
            out_res = out_res * 2

        return blocks

    @property
    def rgb_convs(self):
        rgb_convs = []

        children_dict = {}
        for name, module in self.named_children():
            children_dict[name] = module

        log_out_res = int(np.log2(self.out_res))
        out_res = 4
        for in_log_res in reversed(range(1, log_out_res)):
            out_log_res = in_log_res + 1
            name = '{res}x{res}_from_rgb_lod{lod}'.format(res=out_res, lod=(out_log_res - 2))
            module = children_dict[name]
            rgb_convs.append(module)
            out_res = out_res * 2

        return rgb_convs

    def forward(self, x, labels=None, lod=0, alpha=1):
        blocks = self.blocks
        rgb_convs = self.rgb_convs

        assert 0 <= lod < len(blocks)
        stop_after = len(blocks) - lod - 1

        if stop_after != 0:
            y = self.avgpool(x)
            y = rgb_convs[stop_after - 1](y)
            x = rgb_convs[stop_after](x)
            x = blocks[stop_after](x)
            # x = alpha * x + (1 - alpha) * y
            x = torch.lerp(y, x, alpha)
            stop_after -= 1
        else:
            x = rgb_convs[stop_after](x)

        for i, b in reversed(list(enumerate(blocks))):
            if i <= stop_after:
                x = b(x)

        x = x.view(x.size(0), -1)
        out = self.fc(x)

        if self.use_class_labels:
            if labels.dim() != 2:
                labels = labels.unsqueeze(1)
            out = out.gather(1, labels)
            # index = Variable(torch.LongTensor(range(out.size(0))))
            # if labels.is_cuda:
            # index = index.cuda()
            # out = out[index, labels]
        return out

    def convert_tf_weights_to_state_dict(self, tf_net, device=torch.device('cpu')):
        # full of hacks
        state_dict = {}

        def raise_unexpected(tf_k, v):
            raise RuntimeError("Unexpected key '{}' with shape: {}".format(tf_k, v.size()))

        def transform_conv_w(v):
            # tf: [ k1, k2, nc_in, nc_out ]
            # pt: [ nc_out, nc_in, k1, k2 ]
            return v.permute(3, 2, 0, 1)

        def transform_fc_w(v):
            # tf: [ fan_in, fan_out ]
            # pt: [ fan_out, fan_in ]
            return v.t()

        def from_rgb(k, tf_k, v):
            m = re.match(r'^FromRGB_lod(\d+)\.(weight|bias)$', k)
            if m:
                lod = int(m.group(1))
                k = '{res}x{res}_from_rgb_lod{lod}._{name}'.format(
                    res=int(self.out_res / 2 ** lod), lod=lod, name=m.group(2))
                if m.group(2) == 'weight':
                    v = transform_conv_w(v)
                else:
                    v = v.view(-1)
                return k, v

        def sub_synthesis(k, tf_k, v):
            def replace_tail_if_match(pattern, new):
                nonlocal k
                if k.endswith(pattern):
                    k = k[:-len(pattern)] + new
                    return True
                return False

            if k.startswith('4x4.'):
                if '4x4.Conv' in k:
                    k = k.replace('4x4.Conv', '4x4.Conv0')
                elif '4x4.Dense0' in k:
                    if k == '4x4.Dense0.weight':
                        return '4x4.conv2._weight', transform_fc_w(v)
                    if k == '4x4.Dense0.bias':
                        return '4x4.conv2._bias', v.view(-1)
                    raise_unexpected(tf_k, v)
                elif '4x4.Dense1' in k:
                    k = k.replace('4x4.Dense1', 'fc')
                    if k.endswith('weight'):
                        k = k.replace('weight', '_weight')
                        return k, transform_fc_w(v)
                    if k.endswith('bias'):
                        k = k.replace('bias', '_bias')
                        return k, v
                    raise_unexpected(tf_k, v)
                else:
                    raise_unexpected(tf_k, v)

            if replace_tail_if_match('.Conv0.weight', '.conv1._weight'):  # noqa: E241, E202
                return k, transform_conv_w(v)
            if replace_tail_if_match('.Conv0.bias', '.conv1._bias'):  # noqa: E241, E202
                return k, v.view(-1)

            if 'Conv0' in k:
                raise_unexpected(tf_k, v)

            if replace_tail_if_match('.Conv1_down.weight', '.conv2._weight'):  # noqa: E241, E202
                return k, transform_conv_w(v)
            if replace_tail_if_match('.Conv1_down.bias', '.conv2._bias'):  # noqa: E241, E202
                return k, v.view(-1)

            if 'Conv1' in k:
                raise_unexpected(tf_k, v)

            raise_unexpected(tf_k, v)

        for tf_k, tf_v in tf_net.vars.items():
            assert '.' not in tf_k

            k = tf_k.replace('/', '.')
            v = torch.as_tensor(tf_v.eval())

            if k in {'lod'}:
                # no input buffer
                continue

            elif k.startswith('FromRGB'):
                k, v = from_rgb(k, tf_k, v)

            elif 'weight' in k or 'bias' in k:
                k, v = sub_synthesis(k, tf_k, v)

            else:
                raise_unexpected(tf_k, v)

            state_dict[k] = v

        # tf state dict doesn't have the blur kernels, but pytorch wants to see
        # them
        for k, v in self.state_dict().items():
            if re.match(r'\d+x\d+\.blur\.kernel$', k):
                state_dict[k] = v.detach()
            if re.match(r'\d+x\d+\.pool\.kernel$', k):
                state_dict[k] = v.detach()
            if k == 'avgpool.kernel':
                state_dict[k] = v.detach()

        # device & contiguity
        for k in state_dict:
            state_dict[k] = state_dict[k].to(device).contiguous()

        return state_dict
