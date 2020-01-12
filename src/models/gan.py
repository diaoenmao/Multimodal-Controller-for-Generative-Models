import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_model


def idx2onehot(idx):
    if config.PARAM['subset'] == 'label' or config.PARAM['subset'] == 'identity':
        idx = idx.view(idx.size(0), 1)
        onehot = idx.new_zeros(idx.size(0), config.PARAM['classes_size']).float()
        onehot.scatter_(1, idx, 1)
    else:
        onehot = idx.float()
    return onehot


class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def generate(self, N):
        x = torch.randn([N, config.PARAM['latent_size']], device=config.PARAM['device'])
        generated = self.model['generator'](x)
        generated = generated.view(generated.size(0), *config.PARAM['img_shape'])
        generated = (generated + 1) / 2
        return generated

    def discriminate(self, input):
        x = (input * 2) - 1
        x = x.view(x.size(0), -1)
        x = self.model['discriminator'](x)
        discriminated = x.view(x.size(0))
        return discriminated

    def forward(self, input):
        x = self.generate(input['img'].size(0))
        x = self.discriminate(x)
        return x


class CGAN(nn.Module):
    def __init__(self):
        super(CGAN, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def generate(self, C):
        x = torch.randn([C.size(0), config.PARAM['latent_size']]).to(config.PARAM['device'])
        onehot = idx2onehot(C)
        embedding = self.model['generator_embedding'](onehot)
        x = torch.cat((x, embedding), dim=1)
        generated = self.model['generator'](x)
        generated = generated.view(generated.size(0), *config.PARAM['img_shape'])
        generated = (generated + 1) / 2
        return generated

    def discriminate(self, input, C):
        x = (input * 2) - 1
        x = x.view(x.size(0), -1)
        onehot = idx2onehot(C)
        embedding = self.model['discriminator_embedding'](onehot)
        x = torch.cat((x, embedding), dim=1)
        x = self.model['discriminator'](x)
        discriminated = x.view(x.size(0))
        return discriminated

    def forward(self, input):
        x = self.generate(input[config.PARAM['subset']])
        x = self.discriminate(x, input[config.PARAM['subset']])
        return x


class RMGAN(nn.Module):
    def __init__(self):
        super(RMGAN, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def generate(self, C):
        x = torch.randn([C.size(0), config.PARAM['latent_size']]).to(config.PARAM['device'])
        config.PARAM['attr'] = idx2onehot(C)
        generated = self.model['generator'](x)
        generated = generated.view(generated.size(0), *config.PARAM['img_shape'])
        generated = (generated + 1) / 2
        return generated

    def discriminate(self, input, C):
        x = (input * 2) - 1
        x = x.view(x.size(0), -1)
        config.PARAM['attr'] = idx2onehot(C)
        x = self.model['discriminator'](x)
        discriminated = x.view(x.size(0))
        return discriminated

    def forward(self, input):
        x = self.generate(input[config.PARAM['subset']])
        x = self.discriminate(x, input[config.PARAM['subset']])
        return x


def gan():
    normalization = 'bn1'
    activation = 'leakyrelu'
    img_shape = config.PARAM['img_shape']
    latent_size = config.PARAM['latent_size']
    hidden_size = config.PARAM['hidden_size']
    num_layers_generator = config.PARAM['num_layers_generator']
    num_layers_discriminator = config.PARAM['num_layers_discriminator']
    config.PARAM['model'] = {}
    # Generator
    config.PARAM['model']['generator'] = []
    config.PARAM['model']['generator'].append(
        {'cell': 'LinearCell', 'input_size': latent_size, 'output_size': hidden_size,
         'bias': True, 'normalization': normalization, 'activation': activation})
    for i in range(num_layers_generator - 2):
        config.PARAM['model']['generator'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size * (2 ** i),
             'output_size': hidden_size * (2 ** (i + 1)),
             'bias': True, 'normalization': normalization, 'activation': activation})
    config.PARAM['model']['generator'].append(
        {'cell': 'LinearCell', 'input_size': hidden_size * (2 ** (num_layers_generator - 2)),
         'output_size': np.prod(img_shape), 'bias': True, 'normalization': 'none', 'activation': 'tanh'})
    config.PARAM['model']['generator'] = tuple(config.PARAM['model']['generator'])
    # Discriminator
    config.PARAM['model']['discriminator'] = []
    config.PARAM['model']['discriminator'].append(
        {'cell': 'LinearCell', 'input_size': np.prod(img_shape),
         'output_size': hidden_size * (2 ** (num_layers_discriminator - 1)),
         'bias': True, 'normalization': 'none', 'activation': activation})
    for i in range(num_layers_discriminator - 1):
        config.PARAM['model']['discriminator'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size * (2 ** (num_layers_discriminator - 1)) // (2 ** i),
             'output_size': hidden_size * (2 ** (num_layers_discriminator - 1)) // (2 ** (i + 1)),
             'bias': True, 'normalization': 'none', 'activation': activation})
    config.PARAM['model']['discriminator'].append(
        {'cell': 'LinearCell', 'input_size': hidden_size,
         'output_size': 1, 'bias': True, 'normalization': 'none', 'activation': 'sigmoid'})
    config.PARAM['model']['discriminator'] = tuple(config.PARAM['model']['discriminator'])
    model = GAN()
    return model


def cgan():
    normalization = 'bn1'
    activation = 'leakyrelu'
    img_shape = config.PARAM['img_shape']
    latent_size = config.PARAM['latent_size']
    hidden_size = config.PARAM['hidden_size']
    num_layers_generator = config.PARAM['num_layers_generator']
    num_layers_discriminator = config.PARAM['num_layers_discriminator']
    classes_size = config.PARAM['classes_size']
    config.PARAM['model'] = {}
    # Embedding
    config.PARAM['model']['generator_embedding'] = {
        'cell': 'LinearCell', 'input_size': classes_size, 'output_size': classes_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['discriminator_embedding'] = {
        'cell': 'LinearCell', 'input_size': classes_size, 'output_size': classes_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    # Generator
    config.PARAM['model']['generator'] = []
    config.PARAM['model']['generator'].append(
        {'cell': 'LinearCell', 'input_size': latent_size + classes_size, 'output_size': hidden_size,
         'bias': True, 'normalization': normalization, 'activation': activation})
    for i in range(num_layers_generator - 2):
        config.PARAM['model']['generator'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size * (2 ** i), 'output_size': hidden_size * (2 ** (i + 1)),
             'bias': True, 'normalization': normalization, 'activation': activation})
    config.PARAM['model']['generator'].append(
        {'cell': 'LinearCell', 'input_size': hidden_size * (2 ** (num_layers_generator - 2)),
         'output_size': np.prod(img_shape), 'bias': True, 'normalization': 'none', 'activation': 'tanh'})
    config.PARAM['model']['generator'] = tuple(config.PARAM['model']['generator'])
    # Discriminator
    config.PARAM['model']['discriminator'] = []
    config.PARAM['model']['discriminator'].append(
        {'cell': 'LinearCell', 'input_size': np.prod(img_shape) + classes_size,
         'output_size': hidden_size * (2 ** (num_layers_discriminator - 1)),
         'bias': True, 'normalization': 'none', 'activation': activation})
    for i in range(num_layers_discriminator - 1):
        config.PARAM['model']['discriminator'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size * (2 ** (num_layers_discriminator - 1)) // (2 ** i),
             'output_size': hidden_size * (2 ** (num_layers_discriminator - 1)) // (2 ** (i + 1)),
             'bias': True, 'normalization': 'none', 'activation': activation})
    config.PARAM['model']['discriminator'].append(
        {'cell': 'LinearCell', 'input_size': hidden_size,
         'output_size': 1, 'bias': True, 'normalization': 'none', 'activation': 'sigmoid'})
    config.PARAM['model']['discriminator'] = tuple(config.PARAM['model']['discriminator'])
    model = CGAN()
    return model


def rmgan():
    normalization = 'rbn1'
    activation = 'leakyrelu'
    img_shape = config.PARAM['img_shape']
    latent_size = config.PARAM['latent_size']
    hidden_size = config.PARAM['hidden_size']
    num_layers_generator = config.PARAM['num_layers_generator']
    num_layers_discriminator = config.PARAM['num_layers_discriminator']
    sharing_rate = config.PARAM['sharing_rate']
    num_mode = config.PARAM['classes_size']
    config.PARAM['model'] = {}
    # Generator
    config.PARAM['model']['generator'] = []
    config.PARAM['model']['generator'].append(
        {'cell': 'RLinearCell', 'input_size': latent_size, 'output_size': hidden_size,
         'bias': True, 'sharing_rate': sharing_rate, 'num_mode': num_mode, 'normalization': normalization,
         'activation': activation})
    for i in range(num_layers_generator - 2):
        config.PARAM['model']['generator'].append(
            {'cell': 'RLinearCell', 'input_size': hidden_size * (2 ** i),
             'output_size': hidden_size * (2 ** (i + 1)),
             'bias': True, 'sharing_rate': sharing_rate, 'num_mode': num_mode, 'normalization': normalization,
             'activation': activation})
    config.PARAM['model']['generator'].append(
        {'cell': 'LinearCell', 'input_size': hidden_size * (2 ** (num_layers_generator - 2)),
         'output_size': np.prod(img_shape), 'bias': True, 'normalization': 'none', 'activation': 'tanh'})
    config.PARAM['model']['generator'] = tuple(config.PARAM['model']['generator'])
    # Discriminator
    config.PARAM['model']['discriminator'] = []
    config.PARAM['model']['discriminator'].append(
        {'cell': 'RLinearCell', 'input_size': np.prod(img_shape),
         'output_size': hidden_size * (2 ** (num_layers_discriminator - 1)),
         'bias': True, 'sharing_rate': sharing_rate, 'num_mode': num_mode, 'normalization': 'none',
         'activation': activation})
    for i in range(num_layers_discriminator - 1):
        config.PARAM['model']['discriminator'].append(
            {'cell': 'RLinearCell', 'input_size': hidden_size * (2 ** (num_layers_discriminator - 1)) // (2 ** i),
             'output_size': hidden_size * (2 ** (num_layers_discriminator - 1)) // (2 ** (i + 1)),
             'bias': True, 'sharing_rate': sharing_rate, 'num_mode': num_mode, 'normalization': 'none',
             'activation': activation})
    config.PARAM['model']['discriminator'].append(
        {'cell': 'LinearCell', 'input_size': hidden_size,
         'output_size': 1, 'bias': True, 'normalization': 'none', 'activation': 'sigmoid'})
    config.PARAM['model']['discriminator'] = tuple(config.PARAM['model']['discriminator'])
    model = RMGAN()
    return model


class DCGAN(nn.Module):
    def __init__(self):
        super(DCGAN, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def generate(self, N):
        x = torch.randn([N, config.PARAM['latent_size'], 1, 1], device=config.PARAM['device'])
        generated = self.model['generator'](x)
        generated = (generated + 1) / 2
        return generated

    def discriminate(self, input):
        x = (input * 2) - 1
        x = self.model['discriminator'](x)
        discriminated = x.view(x.size(0))
        return discriminated

    def forward(self, input):
        x = self.generate(input['img'].size(0))
        x = self.discriminate(x)
        return x


class DCCGAN(nn.Module):
    def __init__(self):
        super(DCCGAN, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def generate(self, C):
        x = torch.randn([C.size(0), config.PARAM['latent_size'], 1, 1], device=config.PARAM['device'])
        onehot = idx2onehot(C)
        embedding = self.model['generator_embedding'](onehot)
        embedding = embedding.view(*embedding.size(), 1, 1)
        x = torch.cat((x, embedding), dim=1)
        generated = self.model['generator'](x)
        generated = (generated + 1) / 2
        return generated

    def discriminate(self, input, C):
        x = (input * 2) - 1
        onehot = idx2onehot(C)
        embedding = self.model['discriminator_embedding'](onehot)
        embedding = embedding.view([*embedding.size(), 1, 1]).expand([*embedding.size(), *x.size()[2:]])
        x = torch.cat((x, embedding), dim=1)
        x = self.model['discriminator'](x)
        discriminated = x.view(x.size(0))
        return discriminated

    def forward(self, input):
        x = self.generate(input[config.PARAM['subset']])
        x = self.discriminate(x, input[config.PARAM['subset']])
        return x


def dcgan():
    normalization = 'bn'
    activation = 'leakyrelu'
    img_shape = config.PARAM['img_shape']
    latent_size = config.PARAM['latent_size']
    hidden_size = config.PARAM['hidden_size']
    depth = config.PARAM['depth']
    config.PARAM['model'] = {}
    # Generator
    config.PARAM['model']['generator'] = []
    config.PARAM['model']['generator'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': latent_size,
         'output_size': hidden_size * (2 ** (depth - 1)), 'kernel_size': 4, 'stride': 1, 'padding': 0,
         'bias': False, 'normalization': normalization, 'activation': activation})
    for i in range(depth - 1):
        config.PARAM['model']['generator'].append(
            {'cell': 'ConvTranspose2dCell', 'input_size': hidden_size * (2 ** (depth - 1 - i)),
             'output_size': hidden_size * (2 ** (depth - 1 - i - 1)), 'kernel_size': 4, 'stride': 2, 'padding': 1,
             'bias': False, 'normalization': normalization, 'activation': activation})
    config.PARAM['model']['generator'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': hidden_size,
         'output_size': img_shape[0], 'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False,
         'normalization': 'none', 'activation': 'tanh'})
    config.PARAM['model']['generator'] = tuple(config.PARAM['model']['generator'])
    # Discriminator
    config.PARAM['model']['discriminator'] = []
    config.PARAM['model']['discriminator'].append(
        {'cell': 'Conv2dCell', 'input_size': img_shape[0],
         'output_size': hidden_size, 'kernel_size': 4, 'stride': 2, 'padding': 1,
         'bias': False, 'normalization': 'none', 'activation': activation})
    for i in range(depth - 1):
        config.PARAM['model']['discriminator'].append(
            {'cell': 'Conv2dCell', 'input_size': hidden_size * (2 ** i),
             'output_size': hidden_size * (2 ** (i + 1)), 'kernel_size': 4, 'stride': 2, 'padding': 1,
             'bias': False, 'normalization': normalization, 'activation': activation})
    config.PARAM['model']['discriminator'].append(
        {'cell': 'Conv2dCell', 'input_size': hidden_size * (2 ** (depth - 1)), 'output_size': 1,
         'kernel_size': 4, 'stride': 1, 'padding': 0, 'bias': False, 'normalization': 'none', 'activation': 'sigmoid'})
    config.PARAM['model']['discriminator'] = tuple(config.PARAM['model']['discriminator'])
    model = DCGAN()
    return model


def dccgan():
    normalization = 'bn'
    activation = 'leakyrelu'
    img_shape = config.PARAM['img_shape']
    latent_size = config.PARAM['latent_size']
    hidden_size = config.PARAM['hidden_size']
    depth = config.PARAM['depth']
    classes_size = int(config.PARAM['classes_size'])
    config.PARAM['model'] = {}
    # Embedding
    config.PARAM['model']['generator_embedding'] = {
        'cell': 'LinearCell', 'input_size': classes_size, 'output_size': classes_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['discriminator_embedding'] = {
        'cell': 'LinearCell', 'input_size': classes_size, 'output_size': classes_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    # Generator
    config.PARAM['model']['generator'] = []
    config.PARAM['model']['generator'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': latent_size + classes_size,
         'output_size': hidden_size * (2 ** (depth - 1)), 'kernel_size': 4, 'stride': 1, 'padding': 0,
         'bias': False, 'normalization': normalization, 'activation': activation})
    for i in range(depth - 1):
        config.PARAM['model']['generator'].append(
            {'cell': 'ConvTranspose2dCell', 'input_size': hidden_size * (2 ** (depth - 1 - i)),
             'output_size': hidden_size * (2 ** (depth - 1 - i - 1)), 'kernel_size': 4, 'stride': 2, 'padding': 1,
             'bias': False, 'normalization': normalization, 'activation': activation})
    config.PARAM['model']['generator'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': hidden_size,
         'output_size': img_shape[0], 'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True,
         'normalization': 'none', 'activation': 'tanh'})
    config.PARAM['model']['generator'] = tuple(config.PARAM['model']['generator'])
    # Discriminator
    config.PARAM['model']['discriminator'] = []
    config.PARAM['model']['discriminator'].append(
        {'cell': 'Conv2dCell', 'input_size': img_shape[0] + classes_size,
         'output_size': hidden_size, 'kernel_size': 4, 'stride': 2, 'padding': 1,
         'bias': False, 'normalization': 'none', 'activation': activation})
    for i in range(depth - 1):
        config.PARAM['model']['discriminator'].append(
            {'cell': 'Conv2dCell', 'input_size': hidden_size * (2 ** i),
             'output_size': hidden_size * (2 ** (i + 1)), 'kernel_size': 4, 'stride': 2, 'padding': 1,
             'bias': False, 'normalization': normalization, 'activation': activation})
    config.PARAM['model']['discriminator'].append(
        {'cell': 'Conv2dCell', 'input_size': hidden_size * (2 ** (depth - 1)), 'output_size': 1,
         'kernel_size': 4, 'stride': 1, 'padding': 0, 'bias': False, 'normalization': 'none', 'activation': 'sigmoid'})
    config.PARAM['model']['discriminator'] = tuple(config.PARAM['model']['discriminator'])
    model = DCCGAN()
    return model