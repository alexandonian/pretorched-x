import torch
import numpy as np


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
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

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
        self.forward_backward_bits = (total_bits * 2)
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

        total_megabytes = (total / 8) / (1024**2)
        return total_megabytes, total


class Identity(torch.nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def hook_sizes(model, inputs, verbose=True):

    # Hook the output sizes.
    sizes = []

    def hook_output_size(module, input, output):
        sizes.append(output.shape)

    # Get modules and register forward hooks.
    names, mods = zip(*[(name, p) for name, p in model.named_modules()
                        if list(p.parameters()) and (not p._modules)])
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
