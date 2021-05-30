import torch
import torch.nn as nn
from torch.nn import Parameter as P


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 shape=(1, -1, 1, 1, 1)):
        super().__init__()
        self.shape = shape
        self.mean = P(torch.tensor(mean).view(shape),
                      requires_grad=False)
        self.std = P(torch.tensor(std).view(shape),
                     requires_grad=False)

    def forward(self, x):
        return (x - self.mean) / self.std
