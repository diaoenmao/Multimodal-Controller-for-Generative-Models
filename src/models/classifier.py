import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param

Normalization = nn.BatchNorm2d
Activation = nn.ReLU


def loss(input, output):
    CE = F.cross_entropy(output['label'], input['label'], reduction='mean')
    return CE


class Classifier(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size):
        super().__init__()
        blocks = [
            nn.Conv2d(data_shape[0], hidden_size[0], 3, 1, 1),
            Normalization(hidden_size[0]),
            Activation(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_size[0], hidden_size[1], 3, 1, 1),
            Normalization(hidden_size[1]),
            Activation(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_size[1], hidden_size[2], 3, 1, 1),
            Normalization(hidden_size[2]),
            Activation(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_size[2], hidden_size[3], 3, 1, 1),
            Normalization(hidden_size[3]),
            Activation(inplace=True),
        ]
        self.blocks = nn.Sequential(*blocks)
        self.encoded_shape = [hidden_size[3], data_shape[1] // (2 ** (len(hidden_size) - 1)),
                              data_shape[2] // (2 ** (len(hidden_size) - 1))]
        self.classifier = nn.Linear(np.prod(self.encoded_shape).item(), classes_size)

    def feature(self, input):
        x = input['img']
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        output['label'] = self.classifier(x)
        output['loss'] = loss(input, output)
        return output


def classifier():
    data_shape = cfg['data_shape']
    hidden_size = [8, 16, 32, 64]
    classes_size = cfg['classes_size']
    cfg['model'] = {}
    model = Classifier(data_shape, hidden_size, classes_size)
    model.apply(init_param)
    return model