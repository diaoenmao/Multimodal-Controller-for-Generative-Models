import config
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import make_model, make_SpectralNormalization


class CSNGAN(nn.Module):
    def __init__(self):
        super(CSNGAN, self).__init__()
        self.model = make_model(config.PARAM['model'])
        make_SpectralNormalization(self.model['discriminator'])

    def generate(self, x, C):
        onehot = F.one_hot(C, config.PARAM['classes_size']).float()
        embedding = self.model['generator_embedding'](onehot)
        embedding = embedding.view(*embedding.size(), 1, 1)
        x = torch.cat((x, embedding), dim=1)
        generated = self.model['generator'](x)
        return generated

    def discriminate(self, input, C):
        x = input
        onehot = F.one_hot(C, config.PARAM['classes_size']).float()
        embedding = self.model['discriminator_embedding'](onehot)
        embedding = embedding.view([*embedding.size(), 1, 1]).expand([*embedding.size(), *x.size()[2:]])
        x = torch.cat((x, embedding), dim=1)
        x = self.model['discriminator'](x)
        discriminated = x.view(x.size(0))
        return discriminated

    def forward(self, input):
        x = self.generate(input['label'])
        x = self.discriminate(x, input['label'])
        return x


class MCSNGAN(nn.Module):
    def __init__(self):
        super(MCSNGAN, self).__init__()
        self.model = make_model(config.PARAM['model'])
        make_SpectralNormalization(self.model['discriminator'])
        
    def generate(self, x, C):
        config.PARAM['indicator'] = F.one_hot(C, config.PARAM['classes_size']).float()
        generated = self.model['generator'](x)
        return generated

    def discriminate(self, input, C):
        x = input
        config.PARAM['indicator'] = F.one_hot(C, config.PARAM['classes_size']).float()
        x = self.model['discriminator'](x)
        discriminated = x.view(x.size(0))
        return discriminated

    def forward(self, input):
        x = self.generate(input['label'])
        x = self.discriminate(x, input['label'])
        return x


def csngan():
    generator_normalization = 'bn'
    generator_activation = 'relu'
    discriminator_normalization = 'none'
    discriminator_activation = 'leakyrelu'
    img_shape = config.PARAM['img_shape']
    num_mode = config.PARAM['classes_size']
    latent_size = config.PARAM['latent_size']
    encode_shape = config.PARAM['encode_shape']
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
    output_size = np.prod(encode_shape)
    config.PARAM['model']['generator'] = []
    config.PARAM['model']['generator'].append(
        {'cell': 'LinearCell', 'input_size': input_size, 'output_size': output_size, 'bias': True,
         'normalization': generator_normalization, 'activation': generator_activation})
    config.PARAM['model']['generator'].append({'cell': 'ResizeCell', 'resize': encode_shape})
    input_size = generator_hidden_size
    output_size = generator_hidden_size
    for i in range(3):
        config.PARAM['model']['generator'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'kernel_size': 3,
             'stride': 1, 'padding': 1, 'bias': True, 'normalization': generator_normalization,
             'activation': generator_activation, 'interpolate': 2})
    input_size = generator_hidden_size
    output_size = img_shape[0]
    config.PARAM['model']['generator'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': 'none', 'activation': 'tanh'})
    config.PARAM['model']['generator'] = tuple(config.PARAM['model']['generator'])
    # Discriminator
    config.PARAM['model']['discriminator'] = []
    input_size = img_shape[0] + conditional_embedding_size
    output_size = discriminator_hidden_size
    for i in range(2):
        config.PARAM['model']['discriminator'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'kernel_size': 3,
             'stride': 1, 'padding': 1, 'bias': True, 'normalization': discriminator_normalization,
             'activation': discriminator_activation, 'interpolate': 0.5})
        input_size = output_size
    for i in range(2):
        config.PARAM['model']['discriminator'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'kernel_size': 3,
             'stride': 1, 'padding': 1, 'bias': True, 'normalization': discriminator_normalization,
             'activation': discriminator_activation})
    input_size = discriminator_hidden_size
    output_size = 1
    config.PARAM['model']['discriminator'].append(
        {'cell': 'LinearCell', 'input_size': input_size, 'output_size': output_size, 'bias': True,
         'normalization': 'none', 'activation': 'sigmoid'})
    config.PARAM['model']['discriminator'] = tuple(config.PARAM['model']['discriminator'])
    model = CSNGAN()
    return model


def mcsngan():
    config.PARAM['model'] = {}
    generator_normalization = 'bn'
    generator_activation = 'relu'
    discriminator_normalization = 'none'
    discriminator_activation = 'leakyrelu'
    img_shape = config.PARAM['img_shape']
    num_mode = config.PARAM['classes_size']
    controller_rate = config.PARAM['controller_rate']
    latent_size = config.PARAM['latent_size']
    encode_shape = config.PARAM['encode_shape']
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
    output_size = np.prod(encode_shape)
    config.PARAM['model']['generator'] = []
    config.PARAM['model']['generator'].append(
        {'cell': 'LinearCell', 'input_size': input_size, 'output_size': output_size, 'bias': True,
         'normalization': generator_normalization, 'activation': generator_activation})
    config.PARAM['model']['generator'].append({'cell': 'ResizeCell', 'resize': encode_shape})
    input_size = generator_hidden_size
    output_size = generator_hidden_size
    config.PARAM['model']['generator'].append(
        {'cell': 'MultimodalController', 'input_size': input_size, 'num_mode': num_mode,
         'controller_rate': controller_rate})
    for i in range(3):
        config.PARAM['model']['generator'].append(
            {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'kernel_size': 3,
             'stride': 1, 'padding': 1, 'bias': True, 'normalization': generator_normalization,
             'activation': generator_activation, 'interpolate': 2})
    input_size = generator_hidden_size
    output_size = img_shape[0]
    config.PARAM['model']['generator'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': 'none', 'activation': 'tanh'})
    config.PARAM['model']['generator'] = tuple(config.PARAM['model']['generator'])
    # Discriminator
    config.PARAM['model']['discriminator'] = []
    input_size = img_shape[0] + conditional_embedding_size
    output_size = discriminator_hidden_size
    for i in range(2):
        config.PARAM['model']['discriminator'].append(
            {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'kernel_size': 3,
             'stride': 1, 'padding': 1, 'bias': True, 'normalization': discriminator_normalization,
             'activation': discriminator_activation, 'interpolate': 0.5})
    input_size = output_size
    for i in range(2):
        config.PARAM['model']['discriminator'].append(
            {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'kernel_size': 3,
             'stride': 1, 'padding': 1, 'bias': True, 'normalization': discriminator_normalization,
             'activation': discriminator_activation})
    input_size = discriminator_hidden_size
    output_size = 1
    config.PARAM['model']['discriminator'].append(
        {'cell': 'LinearCell', 'input_size': input_size, 'output_size': output_size, 'bias': True,
         'normalization': 'none', 'activation': 'sigmoid'})
    config.PARAM['model']['discriminator'] = tuple(config.PARAM['model']['discriminator'])
    model = MCSNGAN()
    return model