from .fbresnet import pretrained_settings as fbresnet_settings
from .bninception import pretrained_settings as bninception_settings
from .resnext import pretrained_settings as resnext_settings
from .inceptionv4 import pretrained_settings as inceptionv4_settings
from .inceptionresnetv2 import pretrained_settings as inceptionresnetv2_settings
from .torchvision_models import pretrained_settings as torchvision_models_settings
from .nasnet_mobile import pretrained_settings as nasnet_mobile_settings
from .nasnet import pretrained_settings as nasnet_settings
from .dpn import pretrained_settings as dpn_settings
from .xception import pretrained_settings as xception_settings
from .senet import pretrained_settings as senet_settings
from .cafferesnet import pretrained_settings as cafferesnet_settings
from .pnasnet import pretrained_settings as pnasnet_settings
from .polynet import pretrained_settings as polynet_settings

from .resnet3D import pretrained_settings as resnet3d_settings
from .resnext3D import pretrained_settings as resnext3d_settings

all_settings = [
    fbresnet_settings,
    bninception_settings,
    resnext_settings,
    inceptionv4_settings,
    inceptionresnetv2_settings,
    torchvision_models_settings,
    nasnet_mobile_settings,
    nasnet_settings,
    dpn_settings,
    xception_settings,
    senet_settings,
    cafferesnet_settings,
    pnasnet_settings,
    polynet_settings,
    resnet3d_settings,
    resnext3d_settings,
]

model_names = []
pretrained_settings = {}
for settings in all_settings:
    for model_name, model_settings in settings.items():
        pretrained_settings[model_name] = model_settings
        model_names.append(model_name)

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
