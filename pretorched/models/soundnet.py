from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

__all__ = [
    'BranchedSoundNet', 'SoundNet', 'soundnet8'
]

model_urls = {
    'imagenet': defaultdict(lambda: None, {
        'soundnet8': 'http://pretorched-x.csail.mit.edu/models/soundnet8-8f784005.pth',
    })}


class BranchedSoundNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 64, stride=2, padding=32)
        self.pool1 = nn.MaxPool1d(8, stride=1, padding=0)
        self.conv2 = nn.Conv1d(16, 32, 32, stride=2, padding=16)
        self.pool2 = nn.MaxPool1d(8, stride=1, padding=0)
        self.conv3 = nn.Conv1d(32, 64, 16, stride=2, padding=8)
        self.conv4 = nn.Conv1d(64, 128, 8, stride=2, padding=4)
        self.conv5 = nn.Conv1d(128, 256, 4, stride=2, padding=2)
        self.pool5 = nn.MaxPool1d(4, stride=1, padding=0)
        self.conv6 = nn.Conv1d(256, 512, 4, stride=2, padding=2)
        self.conv7 = nn.Conv1d(512, 1024, 4, stride=2, padding=2)
        self.conv8_1 = nn.Conv1d(1024, 1000, 8, stride=2, padding=0)
        self.conv8_2 = nn.Conv1d(1024, 401, 8, stride=2, padding=0)
        self.last_linear1 = nn.Linear(262000, 1000)
        self.last_linear2 = nn.Linear(105062, 365)
        self.flatten = nn.Flatten()

    def forward(self, input_wav):
        x = self.pool1(F.relu(nn.BatchNorm1d(16)(self.conv1(input_wav))))
        x = self.pool2(F.relu(nn.BatchNorm1d(32)(self.conv2(x))))
        x = F.relu(nn.BatchNorm1d(64)(self.conv3(x)))
        x = F.relu(nn.BatchNorm1d(128)(self.conv4(x)))
        x = self.pool5(F.relu(nn.BatchNorm1d(256)(self.conv5(x))))
        x = F.relu(nn.BatchNorm1d(512)(self.conv6(x)))
        x = F.relu(nn.BatchNorm1d(1024)(self.conv7(x)))
        x_object = self.flatten(F.relu(self.conv8_1(x)))
        x_place = self.flatten(F.relu(self.conv8_2(x)))
        x_object = self.last_linear1(x_object)
        x_place = self.last_linear2(x_place)
        y = [x_object, x_place]
        return y


class SoundNet(nn.Module):
    def __init__(self, num_classes=1000, feature_dim=262000):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv1d(1, 16, 64, stride=2, padding=32)
        self.pool1 = nn.MaxPool1d(8, stride=1, padding=0)
        self.conv2 = nn.Conv1d(16, 32, 32, stride=2, padding=16)
        self.pool2 = nn.MaxPool1d(8, stride=1, padding=0)
        self.conv3 = nn.Conv1d(32, 64, 16, stride=2, padding=8)
        self.conv4 = nn.Conv1d(64, 128, 8, stride=2, padding=4)
        self.conv5 = nn.Conv1d(128, 256, 4, stride=2, padding=2)
        self.pool5 = nn.MaxPool1d(4, stride=1, padding=0)
        self.conv6 = nn.Conv1d(256, 512, 4, stride=2, padding=2)
        self.conv7 = nn.Conv1d(512, 1024, 4, stride=2, padding=2)
        self.conv8 = nn.Conv1d(1024, 1000, 8, stride=2, padding=0)
        self.last_linear = nn.Linear(feature_dim, num_classes)
        self.flatten = nn.Flatten()
        self.fdim = feature_dim

    def features(self, x):
        x = self.pool1(F.relu(nn.BatchNorm1d(16)(self.conv1(x))))
        x = self.pool2(F.relu(nn.BatchNorm1d(32)(self.conv2(x))))
        x = F.relu(nn.BatchNorm1d(64)(self.conv3(x)))
        x = F.relu(nn.BatchNorm1d(128)(self.conv4(x)))
        x = self.pool5(F.relu(nn.BatchNorm1d(256)(self.conv5(x))))
        x = F.relu(nn.BatchNorm1d(512)(self.conv6(x)))
        x = F.relu(nn.BatchNorm1d(1024)(self.conv7(x)))
        x = self.flatten(F.relu(self.conv8(x)))
        return x

    def forward(self, x):
        x = self.features(x)
        return torch.stack([self.last_linear(i)
                            for i in x.split(self.fdim, -1)[:-1] +
                            (x[..., -self.fdim:],)]).mean(0)


def soundnet8(num_classes=1000, pretrained='imagenet'):
    model = SoundNet(num_classes=num_classes)
    if pretrained is not None:
        url = model_urls[pretrained]['soundnet8']
        model.load_state_dict(model_zoo.load_url(url))
    return model
