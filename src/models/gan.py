import config
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import make_model, make_SpectralNormalization


class CGAN(nn.Module):
    def __init__(self):
        super(CGAN, self).__init__()
        self.model = make_model(config.PARAM['model'])
        self.model['discriminator'].apply(make_SpectralNormalization)

    def generate(self, C, x=None):
        if x is None:
            x = torch.randn([C.size(0), config.PARAM['latent_size']], device=config.PARAM['device'])
        onehot = F.one_hot(C, config.PARAM['classes_size']).float()
        embedding = self.model['generator_embedding'](onehot)
        embedding = embedding.view([*embedding.size(), 1, 1])
        x = x.view([*x.size(), 1, 1])
        x = torch.cat((x, embedding), dim=1)
        generated = self.model['generator'](x)
        return generated

    def discriminate(self, x, C):
        onehot = F.one_hot(C, config.PARAM['classes_size']).float()
        embedding = self.model['discriminator_embedding'](onehot)
        embedding = embedding.view([*embedding.size(), 1, 1]).expand([*embedding.size(), *x.size()[2:]])
        x = torch.cat((x, embedding), dim=1)
        discriminated = self.model['discriminator'](x)
        return discriminated

    def forward(self, input):
        x = torch.randn(input['img'].size(0), config.PARAM['latent_size'], device=config.PARAM['device'])
        x = self.generate(x, input['label'])
        x = self.discriminate(x, input['label'])
        return x


class MCGAN(nn.Module):
    def __init__(self):
        super(MCGAN, self).__init__()
        self.model = make_model(config.PARAM['model'])
        self.model['discriminator'].apply(make_SpectralNormalization)

    def generate(self, C, x=None):
        if x is None:
            x = torch.randn([C.size(0), config.PARAM['latent_size']], device=config.PARAM['device'])
        config.PARAM['indicator'] = F.one_hot(C, config.PARAM['classes_size']).float()
        x = x.view([*x.size(), 1, 1])
        generated = self.model['generator'](x)
        return generated

    def discriminate(self, x, C):
        config.PARAM['indicator'] = F.one_hot(C, config.PARAM['classes_size']).float()
        discriminated = self.model['discriminator'](x)
        return discriminated

    def forward(self, input):
        x = torch.randn(input['img'].size(0), config.PARAM['latent_size'], device=config.PARAM['device'])
        x = self.generate(x, input['label'])
        x = self.discriminate(x, input['label'])
        return x


def cgan():
    config.PARAM['model'] = {}
    generator_activation = config.PARAM['generator_activation']
    discriminator_activation = config.PARAM['discriminator_activation']
    generator_normalization = config.PARAM['generator_normalization']
    discriminator_normalization = config.PARAM['discriminator_normalization']
    img_shape = config.PARAM['img_shape']
    num_mode = config.PARAM['classes_size']
    latent_size = config.PARAM['latent_size']
    generator_hidden_size = config.PARAM['generator_hidden_size']
    discriminator_hidden_size = config.PARAM['discriminator_hidden_size']
    conditional_embedding_size = config.PARAM['conditional_embedding_size']
    config.PARAM['model'] = {}
    # Embedding
    config.PARAM['model']['generator_embedding'] = {
        'cell': 'LinearCell', 'input_size': num_mode, 'output_size': conditional_embedding_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['discriminator_embedding'] = {
        'cell': 'LinearCell', 'input_size': num_mode, 'output_size': conditional_embedding_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    # Generator
    input_size = latent_size + conditional_embedding_size
    output_size = generator_hidden_size[0]
    config.PARAM['model']['generator'] = []
    config.PARAM['model']['generator'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size, 'kernel_size': 4,
         'stride': 1, 'padding': 0, 'bias': True, 'normalization': generator_normalization,
         'activation': generator_activation})
    for i in range(1, 4):
        input_size = generator_hidden_size[i - 1]
        output_size = generator_hidden_size[i]
        config.PARAM['model']['generator'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
             'normalization': generator_normalization, 'activation': generator_activation, 'mode': 'up'})
    input_size = generator_hidden_size[-1]
    output_size = img_shape[0]
    config.PARAM['model']['generator'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size, 'normalization': 'none',
         'activation': 'tanh'})
    config.PARAM['model']['generator'] = tuple(config.PARAM['model']['generator'])
    # Discriminator
    config.PARAM['model']['discriminator'] = []
    input_size = img_shape[0] + conditional_embedding_size
    output_size = discriminator_hidden_size
    for i in range(2):
        config.PARAM['model']['discriminator'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
             'normalization': discriminator_normalization, 'activation': discriminator_activation, 'mode': 'down'})
        input_size = output_size
    for i in range(2):
        config.PARAM['model']['discriminator'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
             'normalization': discriminator_normalization, 'activation': discriminator_activation})
    config.PARAM['model']['discriminator'].append({'nn': 'nn.AdaptiveAvgPool2d(1)'})
    config.PARAM['model']['discriminator'].append({'cell': 'ResizeCell', 'resize': [-1]})
    input_size = discriminator_hidden_size
    output_size = 1
    config.PARAM['model']['discriminator'].append(
        {'cell': 'LinearCell', 'input_size': input_size, 'output_size': output_size, 'bias': True,
         'normalization': 'none', 'activation': 'none'})
    config.PARAM['model']['discriminator'] = tuple(config.PARAM['model']['discriminator'])
    model = CGAN()
    return model


def mcgan():
    config.PARAM['model'] = {}
    generator_activation = config.PARAM['generator_activation']
    discriminator_activation = config.PARAM['discriminator_activation']
    generator_normalization = config.PARAM['generator_normalization']
    discriminator_normalization = config.PARAM['discriminator_normalization']
    img_shape = config.PARAM['img_shape']
    num_mode = config.PARAM['classes_size']
    controller_rate = config.PARAM['controller_rate']
    latent_size = config.PARAM['latent_size']
    generator_hidden_size = config.PARAM['generator_hidden_size']
    discriminator_hidden_size = config.PARAM['discriminator_hidden_size']
    config.PARAM['model'] = {}
    # Generator
    input_size = latent_size
    output_size = generator_hidden_size[0]
    config.PARAM['model']['generator'] = []
    config.PARAM['model']['generator'].append(
        {'cell': 'MCConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size, 'kernel_size': 4,
         'stride': 1, 'padding': 0, 'bias': True, 'normalization': generator_normalization,
         'activation': generator_activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    for i in range(1, 4):
        input_size = generator_hidden_size[i - 1]
        output_size = generator_hidden_size[i]
        config.PARAM['model']['generator'].append(
            {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size,
             'hidden_size': output_size, 'normalization': generator_normalization, 'activation': generator_activation,
             'mode': 'up', 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = generator_hidden_size[-1]
    output_size = img_shape[0]
    config.PARAM['model']['generator'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size, 'normalization': 'none',
         'activation': 'tanh'})
    config.PARAM['model']['generator'] = tuple(config.PARAM['model']['generator'])
    # Discriminator
    config.PARAM['model']['discriminator'] = []
    input_size = img_shape[0]
    output_size = discriminator_hidden_size
    for i in range(2):
        config.PARAM['model']['discriminator'].append(
            {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size,
             'hidden_size': output_size, 'normalization': discriminator_normalization,
             'activation': discriminator_activation, 'mode': 'down', 'num_mode': num_mode,
             'controller_rate': controller_rate})
        input_size = output_size
    for i in range(2):
        config.PARAM['model']['discriminator'].append(
            {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size,
             'hidden_size': output_size, 'normalization': discriminator_normalization,
             'activation': discriminator_activation, 'mode': 'pass', 'num_mode': num_mode,
             'controller_rate': controller_rate})
    config.PARAM['model']['discriminator'].append({'nn': 'nn.AdaptiveAvgPool2d(1)'})
    config.PARAM['model']['discriminator'].append({'cell': 'ResizeCell', 'resize': [-1]})
    input_size = discriminator_hidden_size
    output_size = 1
    config.PARAM['model']['discriminator'].append(
        {'cell': 'LinearCell', 'input_size': input_size, 'output_size': output_size, 'bias': True,
         'normalization': 'none', 'activation': 'none'})
    config.PARAM['model']['discriminator'] = tuple(config.PARAM['model']['discriminator'])
    model = MCGAN()
    return model