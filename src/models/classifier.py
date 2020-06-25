import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import make_model


def loss(input, output):
    CE = F.cross_entropy(output['label'], input['label'], reduction='mean')
    return CE


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = make_model(cfg['model'])

    def feature(self, input):
        x = input['img']
        x = self.model['encoder']((x,))[0]
        x = x.view(x.size(0), -1)
        return x

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        x = self.model['encoder']((x,))[0]
        x = x.view(x.size(0), -1)
        output['label'] = self.model['classifier']((x,))[0]
        output['loss'] = loss(input, output)
        return output


def classifier():
    normalization = 'bn'
    activation = 'relu'
    img_shape = cfg['img_shape']
    hidden_size = [8, 16, 32, 64]
    classes_size = cfg['classes_size']
    encode_shape = [hidden_size[-1], img_shape[1] // (2 ** (len(hidden_size) - 1)),
                    img_shape[2] // (2 ** (len(hidden_size) - 1))]
    cfg['model'] = {}
    # Encoder
    cfg['model']['encoder'] = []
    cfg['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': img_shape[0], 'output_size': hidden_size[0],
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    cfg['model']['encoder'].append(
        {'cell': 'Pool2dCell', 'mode': 'max', 'kernel_size': 2})
    cfg['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': hidden_size[0], 'output_size': hidden_size[1],
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    cfg['model']['encoder'].append(
        {'cell': 'Pool2dCell', 'mode': 'max', 'kernel_size': 2})
    cfg['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': hidden_size[1], 'output_size': hidden_size[2],
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    cfg['model']['encoder'].append(
        {'cell': 'Pool2dCell', 'mode': 'max', 'kernel_size': 2})
    cfg['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': hidden_size[2], 'output_size': hidden_size[3],
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    cfg['model']['encoder'] = tuple(cfg['model']['encoder'])
    # Classifier
    cfg['model']['classifier'] = []
    cfg['model']['classifier'].append(
        {'cell': 'LinearCell', 'input_size': np.prod(encode_shape), 'output_size': classes_size,
         'bias': True, 'normalization': 'none', 'activation': 'none'})
    cfg['model']['classifier'] = tuple(cfg['model']['classifier'])
    model = Classifier()
    return model