import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


# The is a pytorch model translated from a Caffe model.
# Note that I left out any dropout layers
# http://memorability.csail.mit.edu/
# Source for the original model:
# Understanding and Predicting Image Memorability at a Large Scale
# A. Khosla, A. S. Raju, A. Torralba and A. Oliva
# International Conference on Computer Vision (ICCV), 2015
# DOI 10.1109/ICCV.2015.275


__all__ = ['MemNet', 'memnet']

pretrained_settings = {
    'memnet': {
        'lamem': {
            'url': 'http://pretorched-x.csail.mit.edu/models/memnet_lamem-a92fdac2.pth',
            'input_space': 'RGB',
            'input_size': [3, 227, 227],
            'input_range': [0, 1],
            'output_range': [0, 1],
            'output_mean': 0.7626,
            'output_bias': 0.65,
            'output_scale': 2,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
    }
}


class MemNet(nn.Module):
    def __init__(self, output_mean=0.7626, output_bias=0.65, output_scale=2, output_range=[0, 1]):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, 4)
        self.pool = nn.MaxPool2d(3, 2)
        self.norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2, groups=2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1, groups=2)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1, groups=2)
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1)
        self.output_mean = output_mean
        self.output_bias = output_bias
        self.output_scale = output_scale
        self.output_range = output_range

    def forward(self, x):
        out = self.forward_all(x)['fc8']
        out = (out - self.output_mean) * self.output_scale + self.output_bias
        out = out.clamp(*self.output_range)
        return out

    def forward_all(self, x):
        conv1 = F.relu(self.conv1(x))
        pool1 = self.pool(conv1)
        norm1 = self.norm(pool1)
        conv2 = F.relu(self.conv2(norm1))
        pool2 = self.pool(conv2)
        norm2 = self.norm(pool2)
        conv3 = F.relu(self.conv3(norm2))
        conv4 = F.relu(self.conv4(conv3))
        conv5 = F.relu(self.conv5(conv4))
        pool5 = self.pool(conv5)
        # x = x.view(-1, self.num_flat_features(x))
        flattened = pool5.view(-1, self.num_flat_features(pool5))
        fc6 = F.relu(self.fc6(flattened))
        fc7 = F.relu(self.fc7(fc6))
        fc8 = self.fc8(fc7)
        return ({"conv1": conv1,
                 "pool1": pool1,
                 "norm1": norm1,
                 "conv2": conv2,
                 "pool2": pool2,
                 "norm2": norm2,
                 "conv3": conv3,
                 "conv4": conv4,
                 "conv5": conv5,
                 "pool5": pool5,
                 "flattened": flattened,
                 "fc6": fc6,
                 "fc7": fc7,
                 "fc8": fc8})

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def memnet(pretrained='lamem'):
    """Constructs memnet model."""
    model = MemNet()
    if pretrained is not None:
        settings = pretrained_settings['memnet'][pretrained]
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.output_mean = settings['output_mean']
        model.output_bias = settings['output_bias']
        model.output_scale = settings['output_scale']
        model.output_range = settings['output_range']
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model
