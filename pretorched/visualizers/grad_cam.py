from collections import OrderedDict

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F


class _PropagationBase(object):
    def __init__(self, model):
        super().__init__()
        # self.device = next(model.parameters()).device
        self.model = model
        self.image = None

    def _encode_one_hot(self, idx):
        # one_hot = torch.FloatTensor(1, self.preds.size(-1)).zero_()
        one_hot = torch.zeros_like(self.preds)
        one_hot[:, idx] = 1.0
        return one_hot.to(self.device)

    def forward(self, image):
        self.image = image.requires_grad_()
        self.model.zero_grad()
        self.preds = self.model(self.image)
        self.probs = F.softmax(self.preds, dim=1)[0]
        self.prob, self.idx = self.probs.sort(0, True)
        return self.prob, self.idx

    def backward(self, idx):
        one_hot = self._encode_one_hot(idx)
        # print(f'one_hot: {one_hot.shape}')
        # print(f'self.preds: {self.preds.shape}')
        self.preds.backward(gradient=one_hot, retain_graph=True)

    @property
    def device(self):
        return next(self.model.parameters()).device


class BackPropagation(_PropagationBase):
    def generate(self):
        output = self.image.grad.detach().cpu().numpy()
        return output.transpose(0, 2, 3, 1)[0]


class GuidedBackPropagation(BackPropagation):
    def __init__(self, model):
        super().__init__(model)

        def func_b(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0), )

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)


class Deconvolution(BackPropagation):
    def __init__(self, model):
        super().__init__(model)

        def func_b(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_out[0], min=0.0), )

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)


class GradCAM(_PropagationBase):
    def __init__(self, model):
        super().__init__(model)
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.detach()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)

        # print('fmaps', fmaps.shape)
        # print('fmaps[0]', fmaps[0].shape)
        # print('weights', weights.shape)
        o = (fmaps * weights).sum(dim=1)
        gcam = torch.clamp(o, min=0.)
        # print('o', o.shape)
        # gcam = (fmaps[0] * weights[0]).sum(dim=0)
        # gcam = torch.clamp(gcam, min=0.)

        gcam -= gcam.min()
        gcam /= gcam.max()

        return gcam.detach()


def apply_heatmap(gcam, raw_image, strength=0.25, colormap=cv2.COLORMAP_JET):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8((1 - gcam) * 255.0), colormap)
    gcam = strength * gcam.astype(np.float) + (1 - strength) * raw_image.astype(np.float)
    out = gcam / gcam.max() * 255.0
    return out
