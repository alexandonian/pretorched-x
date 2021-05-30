"""PyTorch OpCounter:
Adapted from:
    https://github.com/alexandonian/pretorched-x/
    and
    https://github.com/Lyken17/pytorch-OpCounter:
"""

import time
import logging
from collections import Iterable, defaultdict
from distutils.version import LooseVersion

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

multiply_adds = 1


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
    logger.warning(
        "You are using an old version PyTorch {version}, which THOP is not going to support in the future.".format(
            version=torch.__version__
        )
    )


def hook_sizes(model, inputs, verbose=True):

    # Hook the output sizes.
    sizes = []

    def hook_output_size(module, input, output):
        sizes.append(output.shape)

    # Get modules and register forward hooks.
    names, mods = zip(*[(name, p) for name, p in model.named_modules() if list(p.parameters()) and (not p._modules)])
    for m in mods:
        m.register_forward_hook(hook_output_size)

    # Make forward pass.
    with torch.no_grad():
        output = model(*inputs)

    # Display output, if desired.
    if verbose:
        max_len = max(max([len(n) for n in names]), len('Input'))

        for i, input in enumerate(inputs):
            print(f'Input {i:<{max_len}} has shape: {input.shape}')

        for name, s in zip(names, sizes):
            print(f'Layer {name:<{max_len}} has shape: {s}')

    return output, names, sizes


def profile(model, inputs, custom_ops=None, verbose=True):
    register_hooks = {
        nn.Conv1d: count_convNd,
        nn.Conv2d: count_convNd,
        nn.Conv3d: count_convNd,
        nn.ConvTranspose1d: count_convNd,
        nn.ConvTranspose2d: count_convNd,
        nn.ConvTranspose3d: count_convNd,
        nn.BatchNorm1d: count_bn,
        nn.BatchNorm2d: count_bn,
        nn.BatchNorm3d: count_bn,
        nn.ReLU: zero_ops,
        nn.ReLU6: zero_ops,
        nn.LeakyReLU: count_relu,
        nn.MaxPool1d: zero_ops,
        nn.MaxPool2d: zero_ops,
        nn.MaxPool3d: zero_ops,
        nn.AdaptiveMaxPool1d: zero_ops,
        nn.AdaptiveMaxPool2d: zero_ops,
        nn.AdaptiveMaxPool3d: zero_ops,
        nn.AvgPool1d: count_avgpool,
        nn.AvgPool2d: count_avgpool,
        nn.AvgPool3d: count_avgpool,
        nn.AdaptiveAvgPool1d: count_adap_avgpool,
        nn.AdaptiveAvgPool2d: count_adap_avgpool,
        nn.AdaptiveAvgPool3d: count_adap_avgpool,
        nn.Linear: count_linear,
        nn.Dropout: zero_ops,
        nn.Upsample: count_upsample,
        nn.UpsamplingBilinear2d: count_upsample,
        nn.UpsamplingNearest2d: count_upsample,
        nn.LocalResponseNorm: zero_ops,
        nn.Identity: zero_ops,
    }

    handler_collection = []
    if custom_ops is None:
        custom_ops = {}

    def hook_size(module, input, output):
        module._input_size = input[0].shape
        module._output_size = output.shape

    def hook_time_pre(module, input):
        module._time_start = time.time()

    def hook_time_post(module, input, output):
        module._time_end = time.time()
        module._forward_time = (module._time_end - module._time_start) * 1000

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        if hasattr(m, "total_ops") or hasattr(m, "total_params"):
            logger.warning(
                "Either .total_ops or .total_params is already defined in %s. "
                "Be careful, it might change your code's behavior." % str(m)
            )

        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_muls', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is None:
            if verbose:
                logger.info("THOP has not implemented counting method for ", m)
        else:
            if verbose:
                logger.info("Register FLOP counter for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)
            m.register_forward_hook(hook_size)
            m.register_forward_hook(hook_time_post)
            m.register_forward_pre_hook(hook_time_pre)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    total_ops = 0
    total_muls = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_muls += m.total_muls
        total_params += m.total_params

    total_ops = total_ops.item()
    total_muls = total_muls.item()
    total_params = total_params.item()

    data = defaultdict(list)
    for name, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        data['Layer'].append(name)
        data['Type'].append(type(m).__name__)
        data['Input'].append(tuple(getattr(m, '_input_size', (0,))))
        data['Output'].append(tuple(getattr(m, '_output_size', (0,))))
        # data['Output'].append(tuple(m._output_size))
        # data['Weight'].append(tuple(getattr(m, 'weight', torch.tensor([])).size()))
        if getattr(m, 'weight', None) is not None:
            data['Weight'].append(tuple(getattr(m, 'weight', torch.tensor([])).size()))
        data['Kernel_Size'].append(getattr(m, 'kernel_size', 0))
        data['Padding'].append(getattr(m, 'padding', 0))
        data['Stride'].append(getattr(m, 'stride', 0))
        data['Params'].append(m.total_params.item())
        data['Mults'].append(m.total_muls.item())
        data['Ops'].append(m.total_ops.item())
        data['Time'].append(getattr(m, '_forward_time', 0))

    max_len = {k: max(len(str(x)) for x in v + [k]) for k, v in data.items()}
    totals = {k: sum(v) if k in ['Params', 'Mults', 'Ops', 'Time'] else '' for k, v in data.items()}
    totals['Weight'] = 'Totals:'

    header_str = ''
    for k, v in data.items():
        header_str += f'{k:<{max_len[k]}}\t'

    print(header_str)
    print((len(header_str) + 36) * '-')
    for d in zip(*data.values()):
        line = ''
        for x, ml in zip(d, max_len.values()):
            line += f'{str(x):<{ml}}\t'
        print(line)
    print((len(header_str) + 36) * '-')
    line = ''
    for x, ml in zip(totals.values(), max_len.values()):
        line += f'{str(x):<{ml}}\t'
    print(line)

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()

    # remove temporal buffers
    for n, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        if "total_ops" in m._buffers:
            m._buffers.pop("total_ops")
        if "total_muls" in m._buffers:
            m._buffers.pop("total_muls")
        if "total_params" in m._buffers:
            m._buffers.pop("total_params")
        if hasattr(m, '_input_size'):
            del m._input_size
        if hasattr(m, '_output_size'):
            del m._output_size
        if hasattr(m, '_time_start'):
            del m._time_start
        if hasattr(m, '_time_end'):
            del m._time_end
        if hasattr(m, '_forward'):
            del m._forward_time

    return data


def profile_2(model: nn.Module, inputs, custom_ops=None, verbose=True):
    register_hooks = {
        nn.Conv1d: count_convNd,
        nn.Conv2d: count_convNd,
        nn.Conv3d: count_convNd,
        nn.ConvTranspose1d: count_convNd,
        nn.ConvTranspose2d: count_convNd,
        nn.ConvTranspose3d: count_convNd,
        nn.BatchNorm1d: count_bn,
        nn.BatchNorm2d: count_bn,
        nn.BatchNorm3d: count_bn,
        nn.ReLU: zero_ops,
        nn.ReLU6: zero_ops,
        nn.LeakyReLU: count_relu,
        nn.MaxPool1d: zero_ops,
        nn.MaxPool2d: zero_ops,
        nn.MaxPool3d: zero_ops,
        nn.AdaptiveMaxPool1d: zero_ops,
        nn.AdaptiveMaxPool2d: zero_ops,
        nn.AdaptiveMaxPool3d: zero_ops,
        nn.AvgPool1d: count_avgpool,
        nn.AvgPool2d: count_avgpool,
        nn.AvgPool3d: count_avgpool,
        nn.AdaptiveAvgPool1d: count_adap_avgpool,
        nn.AdaptiveAvgPool2d: count_adap_avgpool,
        nn.AdaptiveAvgPool3d: count_adap_avgpool,
        nn.Linear: count_linear,
        nn.Dropout: zero_ops,
        nn.Upsample: count_upsample,
        nn.UpsamplingBilinear2d: count_upsample,
        nn.UpsamplingNearest2d: count_upsample,
    }

    handler_collection = {}
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m: nn.Module):
        # if hasattr(m, "total_ops") or hasattr(m, "total_params"):
        #     logger.warning("Either .total_ops or .total_params is already defined in %s. "
        #                    "Be careful, it might change your code's behavior." % m._get_name())
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None

        # if defined both op maps, custom_ops takes higher priority.
        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is None:
            if verbose:
                logger.info("THOP has not implemented counting method for %s." % m._get_name())
        else:
            if verbose:
                logger.info("Register FLOP counter for module %s." % m._get_name())
            handler_collection[m] = m.register_forward_hook(fn)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
        total_ops, total_params = 0, 0
        for m in module.children():
            # if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
            #     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            # else:
            #     m_ops, m_params = m.total_ops, m.total_params
            if m in handler_collection:
                m_ops, m_params = m.total_ops, m.total_params
            else:
                m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            total_ops += m_ops
            total_params += m_params
        #  print(prefix, module._get_name(), (total_ops.item(), total_params.item()))
        return total_ops, total_params

    total_ops, total_params = (_.item() for _ in dfs_count(model))

    # reset model to original status
    model.train(training)
    for m, handler in handler_collection.items():
        handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")

    return total_ops, total_params


def zero_ops(m, x, y):
    m.total_ops += torch.Tensor([int(0)])


def count_convNd(m: _ConvNd, x: (torch.Tensor,), y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops + bias_ops)
    total_muls = y.nelement() * (m.in_channels // m.groups * kernel_ops)

    m.total_ops += torch.Tensor([int(total_ops)])
    m.total_muls += torch.Tensor([int(total_muls)])


def count_convNd_ver2(m: _ConvNd, x: (torch.Tensor,), y: torch.Tensor):
    x = x[0]

    # N x H x W (exclude Cout)
    output_size = torch.zeros((y.size()[:1] + y.size()[2:])).numel()
    # Cout x Cin x Kw x Kh
    kernel_ops = m.weight.nelement()
    if m.bias is not None:
        # Cout x 1
        kernel_ops += +m.bias.nelement()
    # x N x H x W x Cout x (Cin x Kw x Kh + bias)
    m.total_ops += torch.Tensor([int(output_size * kernel_ops)])


def count_bn(m, x, y):
    x = x[0]

    nelements = x.numel()
    if not m.training:
        # subtract, divide, gamma, beta
        total_ops = 2 * nelements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()

    m.total_ops += torch.Tensor([int(nelements)])


def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
    # total_add = torch.prod(torch.Tensor([m.kernel_size]))
    # total_div = 1
    # kernel_ops = total_add + total_div
    kernel_ops = 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_adap_avgpool(m, x, y):
    kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


# TODO: verify the accuracy
def count_upsample(m, x, y):
    if m.mode not in ("nearest", "linear", "bilinear", "bicubic",):  # "trilinear"
        logger.warning("mode %s is not implemented yet, take it a zero op" % m.mode)
        return zero_ops(m, x, y)

    if m.mode == "nearest":
        return zero_ops(m, x, y)

    x = x[0]
    if m.mode == "linear":
        total_ops = y.nelement() * 5  # 2 muls + 3 add
    elif m.mode == "bilinear":
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        total_ops = y.nelement() * 11  # 6 muls + 5 adds
    elif m.mode == "bicubic":
        # https://en.wikipedia.org/wiki/Bicubic_interpolation
        # Product matrix [4x4] x [4x4] x [4x4]
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops = y.nelement() * (ops_solve_A + ops_solve_p)
    elif m.mode == "trilinear":
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # can viewed as 2 bilinear + 1 linear
        total_ops = y.nelement() * (13 * 2 + 5)

    m.total_ops += torch.Tensor([int(total_ops)])


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements
    total_muls = total_mul * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])
    m.total_muls += torch.Tensor([int(total_muls)])


def count_lstm(m: nn.LSTM, x, y):
    if m.num_layers > 1 or m.bidirectional:
        raise NotImplementedError("")

    hidden_size = m.hidden_size
    input_size = m.input_size
    # assume the layout is (length, batch, features)
    length = x[0].size(0)
    batch_size = x[0].size(1)

    muls = 0
    '''
        f_t & = \sigma(W_f \cdot z + b_f)
    '''
    muls += hidden_size * (input_size + hidden_size) * batch_size * length
    '''
        i_t & = \sigma(W_i \cdot z + b_i)
    '''
    muls += hidden_size * (input_size + hidden_size) * batch_size * length
    '''
        g_t & = tanh(W_C \cdot z + b_C)
    '''
    muls += hidden_size * (input_size + hidden_size) * batch_size * length
    '''
        o_t & = \sigma(W_o \cdot z + b_t)
    '''
    muls += hidden_size * (input_size + hidden_size) * batch_size * length

    '''
        C_t & = f_t * C_{t-1} + i_t * g_t
    '''
    muls += 2 * hidden_size * batch_size
    '''
        h_t &= o_t * tanh(C_t)
    '''
    muls += hidden_size * 2

    m.total_ops += muls


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums
