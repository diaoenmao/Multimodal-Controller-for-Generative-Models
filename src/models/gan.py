import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import cfg
from .utils import make_model, make_SpectralNormalization


class CGAN(nn.Module):
    def __init__(self):
        super(CGAN, self).__init__()
        self.model = make_model(cfg['model'])
        self.model['discriminator'].apply(make_SpectralNormalization)

    def generate(self, C, x=None):
        if x is None:
            x = torch.randn([C.size(0), cfg['latent_size']], device=cfg['device'])
        onehot = F.one_hot(C, cfg['classes_size']).float()
        embedding = self.model['generator_embedding'](onehot)
        embedding = embedding.view([*embedding.size(), 1, 1])
        x = x.view([*x.size(), 1, 1])
        x = torch.cat((x, embedding), dim=1)
        generated = self.model['generator'](x)
        return generated

    def discriminate(self, x, C):
        onehot = F.one_hot(C, cfg['classes_size']).float()
        embedding = self.model['discriminator_embedding'](onehot)
        embedding = embedding.view([*embedding.size(), 1, 1]).expand([*embedding.size(), *x.size()[2:]])
        x = torch.cat((x, embedding), dim=1)
        discriminated = self.model['discriminator'](x)
        return discriminated

    def forward(self, input):
        x = torch.randn(input['img'].size(0), cfg['latent_size'], device=cfg['device'])
        x = self.generate(input['label'], x)
        x = self.discriminate(x, input['label'])
        return x


class MCGAN(nn.Module):
    def __init__(self):
        super(MCGAN, self).__init__()
        self.model = make_model(cfg['model'])
        self.model['discriminator'].apply(make_SpectralNormalization)

    def generate(self, C, x=None):
        if x is None:
            x = torch.randn([C.size(0), cfg['latent_size']], device=cfg['device'])
        cfg['indicator'] = F.one_hot(C, cfg['classes_size']).float()
        x = x.view([*x.size(), 1, 1])
        generated = self.model['generator'](x)
        return generated

    def discriminate(self, x, C):
        cfg['indicator'] = F.one_hot(C, cfg['classes_size']).float()
        discriminated = self.model['discriminator'](x)
        return discriminated

    def forward(self, input):
        x = torch.randn(input['img'].size(0), cfg['latent_size'], device=cfg['device'])
        x = self.generate(input['label'], x)
        x = self.discriminate(x, input['label'])
        return x


def cgan():
    cfg['model'] = {}
    generator_activation = cfg['generator_activation']
    discriminator_activation = cfg['discriminator_activation']
    generator_normalization = cfg['generator_normalization']
    discriminator_normalization = cfg['discriminator_normalization']
    img_shape = cfg['img_shape']
    num_mode = cfg['classes_size']
    latent_size = cfg['latent_size']
    generator_hidden_size = cfg['generator_hidden_size']
    discriminator_hidden_size = cfg['discriminator_hidden_size']
    conditional_embedding_size = cfg['conditional_embedding_size']
    cfg['model'] = {}
    # Embedding
    cfg['model']['generator_embedding'] = {
        'cell': 'LinearCell', 'input_size': num_mode, 'output_size': conditional_embedding_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    cfg['model']['discriminator_embedding'] = {
        'cell': 'LinearCell', 'input_size': num_mode, 'output_size': conditional_embedding_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    # Generator
    input_size = latent_size + conditional_embedding_size
    output_size = generator_hidden_size[0]
    cfg['model']['generator'] = []
    cfg['model']['generator'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size, 'kernel_size': 4,
         'stride': 1, 'padding': 0, 'bias': True, 'normalization': generator_normalization,
         'activation': generator_activation})
    for i in range(1, 4):
        input_size = generator_hidden_size[i - 1]
        output_size = generator_hidden_size[i]
        cfg['model']['generator'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
             'normalization': generator_normalization, 'activation': generator_activation, 'mode': 'up'})
    input_size = generator_hidden_size[-1]
    output_size = img_shape[0]
    cfg['model']['generator'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size, 'normalization': 'none',
         'activation': 'tanh'})
    cfg['model']['generator'] = tuple(cfg['model']['generator'])
    # Discriminator
    cfg['model']['discriminator'] = []
    input_size = img_shape[0] + conditional_embedding_size
    output_size = discriminator_hidden_size[0]
    cfg['model']['discriminator'].append(
        {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
         'normalization': discriminator_normalization, 'activation': discriminator_activation, 'mode': 'down'})
    input_size = discriminator_hidden_size[0]
    output_size = discriminator_hidden_size[1]
    cfg['model']['discriminator'].append(
        {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
         'normalization': discriminator_normalization, 'activation': discriminator_activation, 'mode': 'down'})
    input_size = discriminator_hidden_size[1]
    output_size = discriminator_hidden_size[2]
    cfg['model']['discriminator'].append(
        {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
         'normalization': discriminator_normalization, 'activation': discriminator_activation, 'mode': 'down'})
    input_size = discriminator_hidden_size[2]
    output_size = discriminator_hidden_size[3]
    cfg['model']['discriminator'].append(
        {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
         'normalization': discriminator_normalization, 'activation': discriminator_activation})
    cfg['model']['discriminator'].append({'nn': 'nn.AdaptiveAvgPool2d(1)'})
    cfg['model']['discriminator'].append({'cell': 'ResizeCell', 'resize': [-1]})
    input_size = discriminator_hidden_size[-1]
    output_size = 1
    cfg['model']['discriminator'].append(
        {'cell': 'LinearCell', 'input_size': input_size, 'output_size': output_size, 'bias': True,
         'normalization': 'none', 'activation': 'none'})
    cfg['model']['discriminator'] = tuple(cfg['model']['discriminator'])
    model = CGAN()
    return model


def mcgan():
    cfg['model'] = {}
    generator_activation = cfg['generator_activation']
    discriminator_activation = cfg['discriminator_activation']
    generator_normalization = cfg['generator_normalization']
    discriminator_normalization = cfg['discriminator_normalization']
    img_shape = cfg['img_shape']
    num_mode = cfg['classes_size']
    controller_rate = cfg['controller_rate']
    latent_size = cfg['latent_size']
    generator_hidden_size = cfg['generator_hidden_size']
    discriminator_hidden_size = cfg['discriminator_hidden_size']
    cfg['model'] = {}
    # Generator
    input_size = latent_size
    output_size = generator_hidden_size[0]
    cfg['model']['generator'] = []
    cfg['model']['generator'].append(
        {'cell': 'MCConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size, 'kernel_size': 4,
         'stride': 1, 'padding': 0, 'bias': True, 'normalization': generator_normalization,
         'activation': generator_activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    for i in range(1, 4):
        input_size = generator_hidden_size[i - 1]
        output_size = generator_hidden_size[i]
        cfg['model']['generator'].append(
            {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size,
             'hidden_size': output_size, 'normalization': generator_normalization, 'activation': generator_activation,
             'mode': 'up', 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = generator_hidden_size[-1]
    output_size = img_shape[0]
    cfg['model']['generator'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size, 'normalization': 'none',
         'activation': 'tanh'})
    cfg['model']['generator'] = tuple(cfg['model']['generator'])
    # Discriminator
    cfg['model']['discriminator'] = []
    input_size = img_shape[0]
    output_size = discriminator_hidden_size[0]
    cfg['model']['discriminator'].append(
        {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size,
         'hidden_size': output_size, 'normalization': discriminator_normalization,
         'activation': discriminator_activation, 'mode': 'down', 'num_mode': num_mode,
         'controller_rate': controller_rate})
    input_size = discriminator_hidden_size[0]
    output_size = discriminator_hidden_size[1]
    cfg['model']['discriminator'].append(
        {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size,
         'hidden_size': output_size, 'normalization': discriminator_normalization,
         'activation': discriminator_activation, 'mode': 'down', 'num_mode': num_mode,
         'controller_rate': controller_rate})
    input_size = discriminator_hidden_size[1]
    output_size = discriminator_hidden_size[2]
    cfg['model']['discriminator'].append(
        {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size,
         'hidden_size': output_size, 'normalization': discriminator_normalization,
         'activation': discriminator_activation, 'mode': 'down', 'num_mode': num_mode,
         'controller_rate': controller_rate})
    input_size = discriminator_hidden_size[2]
    output_size = discriminator_hidden_size[3]
    cfg['model']['discriminator'].append(
        {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size,
         'hidden_size': output_size, 'normalization': discriminator_normalization,
         'activation': discriminator_activation, 'mode': 'pass', 'num_mode': num_mode,
         'controller_rate': controller_rate})
    cfg['model']['discriminator'].append({'nn': 'nn.AdaptiveAvgPool2d(1)'})
    cfg['model']['discriminator'].append({'cell': 'ResizeCell', 'resize': [-1]})
    input_size = discriminator_hidden_size[-1]
    output_size = 1
    cfg['model']['discriminator'].append(
        {'cell': 'LinearCell', 'input_size': input_size, 'output_size': output_size, 'bias': True,
         'normalization': 'none', 'activation': 'none'})
    cfg['model']['discriminator'] = tuple(cfg['model']['discriminator'])
    model = MCGAN()
    return model