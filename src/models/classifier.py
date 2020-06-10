import config
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_model


def loss(input, output):
    CE = F.cross_entropy(output['label'], input['label'], reduction='mean')
    return CE


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def feature(self, input):
        x = input['img']
        x = self.model['encoder'](x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['img']
        x = self.model['encoder'](x)
        x = x.view(x.size(0), -1)
        output['label'] = self.model['classifier'](x)
        output['loss'] = loss(input, output)
        return output


def classifier():
    normalization = 'bn'
    activation = 'relu'
    img_shape = config.PARAM['img_shape']
    hidden_size = [8, 16, 32, 64]
    classes_size = config.PARAM['classes_size']
    encode_shape = [hidden_size[-1], img_shape[1] // (2 ** (len(hidden_size) - 1)),
                    img_shape[2] // (2 ** (len(hidden_size) - 1))]
    config.PARAM['model'] = {}
    # Encoder
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': img_shape[0], 'output_size': hidden_size[0],
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    config.PARAM['model']['encoder'].append(
        {'nn': 'nn.MaxPool2d(2, 2)'})
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': hidden_size[0], 'output_size': hidden_size[1],
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    config.PARAM['model']['encoder'].append(
        {'nn': 'nn.MaxPool2d(2, 2)'})
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': hidden_size[1], 'output_size': hidden_size[2],
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    config.PARAM['model']['encoder'].append(
        {'nn': 'nn.MaxPool2d(2, 2)'})
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': hidden_size[2], 'output_size': hidden_size[3],
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    # Classifier
    config.PARAM['model']['classifier'] = []
    config.PARAM['model']['classifier'].append(
        {'cell': 'LinearCell', 'input_size': np.prod(encode_shape), 'output_size': classes_size,
         'bias': True, 'normalization': 'none', 'activation': 'none'})
    config.PARAM['model']['classifier'] = tuple(config.PARAM['model']['classifier'])
    model = Classifier()
    return model