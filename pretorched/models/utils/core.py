import copy
import re
from collections import OrderedDict
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter as P


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class EMA(object):
    """Apply EMA to a model.

    Simple wrapper that applies EMA to a model. Could be better done in 1.0 using
    the parameters() and buffers() module functions, but for now this works
    with state_dicts using .copy_

    """

    def __init__(self, source, target=None, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target if target is not None else copy.deepcopy(source)
        self.decay = decay

        # Optional parameter indicating what iteration to start the decay at.
        self.start_itr = start_itr

        # Initialize target's params to be source's.
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()

        print('Initializing EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)

    def update(self, itr=None):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(
                    self.target_dict[key].data * decay
                    + self.source_dict[key].data * (1 - decay)
                )

    def __repr__(self):
        return (
            f'Source: {type(self.source).__name__}\n'
            f'Target: {type(self.target).__name__}'
        )


def ortho(model, strength=1e-4, blacklist=[]):
    """Apply modified ortho reg to a model.

    This function is an optimized version that directly computes the gradient,
    instead of computing and then differentiating the loss.
    """
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes, and not in the blacklist.
            if len(param.shape) < 2 or any(param is item for item in blacklist):
                continue
            w = param.view(param.shape[0], -1)
            grad = 2 * torch.mm(
                torch.mm(w, w.t()) * (1.0 - torch.eye(w.shape[0], device=w.device)), w
            )
            param.grad.data += strength * grad.view(param.shape)


def default_ortho(model, strength=1e-4, blacklist=[]):
    """Default ortho regularization.

    This function is an optimized version that directly computes the gradient,
    instead of computing and then differentiating the loss.
    """
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes & not in blacklist.
            if len(param.shape) < 2 or param in blacklist:
                continue
            w = param.view(param.shape[0], -1)
            grad = 2 * torch.mm(
                torch.mm(w, w.t()) - torch.eye(w.shape[0], device=w.device), w
            )
            param.grad.data += strength * grad.view(param.shape)


def hook_sizes(model, inputs, verbose=True) -> Tuple[Any, List[str], List[torch.Size]]:

    # Hook the output sizes.
    sizes = []

    def hook_output_size(module, input, output):
        sizes.append(output.shape)

    # Get modules and register forward hooks.
    names, mods = zip(
        *[
            (name, p)
            for name, p in model.named_modules()
            if list(p.parameters()) and (not p._modules)
        ]
    )
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


class SizeEstimator(object):
    def __init__(self, model, input_size=(1, 1, 32, 32), bits=32):
        """Estimates the size of PyTorch models in memory for a given input size."""
        self.model = model
        self.input_size = input_size
        self.bits = 32
        return

    def get_parameter_sizes(self):
        """Get sizes of all parameters in `model`."""
        mods = list(self.model.modules())
        sizes = []

        for i in range(1, len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for item in p:
                sizes.append(np.array(item.size()))

        self.param_sizes = sizes
        return

    def get_output_sizes(self):
        """Run sample input through each layer to get output sizes."""
        input_ = torch.FloatTensor(*self.input_size)
        mods = list(self.model.modules())
        out_sizes = []
        for i in range(1, len(mods)):
            m = mods[i]
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

        self.out_sizes = out_sizes
        return

    def calc_param_bits(self):
        """Calculate total number of bits to store `model` parameters."""
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def calc_forward_backward_bits(self):
        """Calculate bits to store forward and backward pass."""
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = total_bits * 2
        return

    def calc_input_bits(self):
        """Calculate bits to store input."""
        self.input_bits = np.prod(np.array(self.input_size)) * self.bits
        return

    def estimate_size(self):
        """Estimate model size in memory in megabytes and bits."""
        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits + self.input_bits

        total_megabytes = (total / 8) / (1024 ** 2)
        return total_megabytes, total


def remove_prefix(state_dict, prefix='module.'):
    return {re.sub(fr'^{prefix}', '', k): v for k, v in state_dict.items()}


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = pretorched.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = pretorched.models.utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


def elastic_forward(model, input):
    error_msg = 'CUDA out of memory.'

    def chunked_forward(f, x, chunk_size=1):
        return torch.cat([f(xc.contiguous()).detach() for xc in x.chunk(chunk_size)])

    # def _chunked_forward(f, x, chunk_size=1):
    #     out = []
    #     for xc in torch.chunk(x, chunk_size):
    #         print(xc.shape)
    #         o = f(xc).detach().cpu()
    #         out.append(o)
    #     return torch.cat(out)
    # return torch.cat([f(xc.contiguous()) for xc in torch.chunk(x, chunk_size)])

    cs, fit = 1, False
    while not fit:
        try:
            print(f'chunk_size: {cs}')
            return chunked_forward(model, input.contiguous(), cs)
        except RuntimeError as e:
            print(str(e))
            if error_msg in str(e):
                print('| WARNING: ran out of memory, retrying batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                # cs += 1
                cs *= 2
            else:
                raise e


class Normalize(nn.Module):
    def __init__(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        shape=(1, -1, 1, 1, 1),
    ):
        super().__init__()
        self.shape = shape
        self.mean = P(torch.tensor(mean).view(shape), requires_grad=False)
        self.std = P(torch.tensor(std).view(shape), requires_grad=False)

    def forward(self, x):
        return (x - self.mean) / self.std
