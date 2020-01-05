import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_model


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def loss(input, output):
    CE = F.binary_cross_entropy(output['img'], input['img'], reduction='sum')
    KLD = -0.5 * torch.sum(1 + output['logvar'] - output['mu'].pow(2) - output['logvar'].exp())
    return CE + KLD


def idx2onehot(idx):
    if config.PARAM['subset'] == 'label' or config.PARAM['subset'] == 'identity':
        idx = idx.view(idx.size(0), 1)
        onehot = idx.new_zeros(idx.size(0), config.PARAM['classes_size']).float()
        onehot.scatter_(1, idx, 1)
    else:
        onehot = idx.float()
    return onehot


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def generate(self, N):
        x = torch.randn([N, config.PARAM['latent_size']], device=config.PARAM['device'])
        x = self.model['decoder_latent'](x)
        x = self.model['decoder'](x)
        generated = x.view(x.size(0), *config.PARAM['img_shape'])
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['img']
        x = x.view(x.size(0), -1)
        x = self.model['encoder'](x)
        x = self.model['encoder_latent'](x)
        output['mu'], output['logvar'] = torch.chunk(x, 2, dim=1)
        if self.training:
            x = reparameterize(output['mu'], output['logvar'])
        else:
            x = output['mu']
        x = self.model['decoder_latent'](x)
        decoded = self.model['decoder'](x)
        output['img'] = decoded.view(decoded.size(0), *config.PARAM['img_shape'])
        output['loss'] = loss(input, output)
        return output


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def generate(self, C):
        x = torch.randn([C.size(0), config.PARAM['latent_size']], device=config.PARAM['device'])
        onehot = idx2onehot(C)
        decoder_embedding = self.model['decoder_embedding'](onehot)
        x = torch.cat((x, decoder_embedding), dim=1)
        x = self.model['decoder_latent'](x)
        x = self.model['decoder'](x)
        generated = x.view(x.size(0), *config.PARAM['img_shape'])
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['img']
        x = x.view(x.size(0), -1)
        onehot = idx2onehot(input[config.PARAM['subset']])
        encoder_embedding = self.model['encoder_embedding'](onehot)
        x = torch.cat((x, encoder_embedding), dim=1)
        x = self.model['encoder'](x)
        x = self.model['encoder_latent'](x)
        output['mu'], output['logvar'] = torch.chunk(x, 2, dim=1)
        if self.training:
            x = reparameterize(output['mu'], output['logvar'])
        else:
            x = output['mu']
        decoder_embedding = self.model['decoder_embedding'](onehot)
        x = torch.cat((x, decoder_embedding), dim=1)
        x = self.model['decoder_latent'](x)
        decoded = self.model['decoder'](x)
        output['img'] = decoded.view(decoded.size(0), *config.PARAM['img_shape'])
        output['loss'] = loss(input, output)
        return output


def vae():
    normalization = 'bn1'
    activation = 'relu'
    img_shape = config.PARAM['img_shape']
    latent_size = config.PARAM['latent_size']
    hidden_size = config.PARAM['hidden_size']
    num_layers = config.PARAM['num_layers']
    config.PARAM['model'] = {}
    # Encoder
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'LinearCell', 'input_size': np.prod(img_shape), 'output_size': hidden_size,
         'bias': True, 'normalization': normalization, 'activation': activation})
    for i in range(num_layers - 2):
        config.PARAM['model']['encoder'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': hidden_size, 'bias': True,
             'normalization': normalization, 'activation': activation})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    # latent
    config.PARAM['model']['encoder_latent'] = {
        'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': 2 * latent_size, 'bias': True,
        'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['decoder_latent'] = {
        'cell': 'LinearCell', 'input_size': latent_size, 'output_size': hidden_size, 'bias': True,
        'normalization': 'none', 'activation': activation}
    # Decoder
    config.PARAM['model']['decoder'] = []
    for i in range(num_layers - 2):
        config.PARAM['model']['decoder'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': hidden_size, 'bias': True,
             'normalization': normalization, 'activation': activation})
    config.PARAM['model']['decoder'].append(
        {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': np.prod(img_shape),
         'bias': True, 'normalization': 'none', 'activation': 'sigmoid'})
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = VAE()
    return model


def cvae():
    normalization = 'bn1'
    activation = 'relu'
    img_shape = config.PARAM['img_shape']
    latent_size = config.PARAM['latent_size']
    hidden_size = config.PARAM['hidden_size']
    num_layers = config.PARAM['num_layers']
    classes_size = config.PARAM['classes_size']
    config.PARAM['model'] = {}
    # Embedding
    config.PARAM['model']['encoder_embedding'] = {
        'cell': 'LinearCell', 'input_size': classes_size, 'output_size': classes_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['decoder_embedding'] = {
        'cell': 'LinearCell', 'input_size': classes_size, 'output_size': classes_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    # Encoder
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'LinearCell', 'input_size': np.prod(img_shape) + classes_size,
         'output_size': hidden_size, 'bias': True, 'normalization': normalization, 'activation': activation})
    for i in range(num_layers - 2):
        config.PARAM['model']['encoder'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': hidden_size, 'bias': True,
             'normalization': normalization, 'activation': activation})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    # latent
    config.PARAM['model']['encoder_latent'] = {
        'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': 2 * latent_size, 'bias': True,
        'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['decoder_latent'] = {
        'cell': 'LinearCell', 'input_size': latent_size + classes_size, 'output_size': hidden_size, 'bias': True,
        'normalization': 'none', 'activation': activation}
    # Decoder
    config.PARAM['model']['decoder'] = []
    for i in range(num_layers - 2):
        config.PARAM['model']['decoder'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': hidden_size,
             'bias': True, 'normalization': normalization, 'activation': activation})
    config.PARAM['model']['decoder'].append(
        {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': np.prod(img_shape),
         'bias': True, 'normalization': 'none', 'activation': 'sigmoid'})
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = CVAE()
    return model


class DCVAE(nn.Module):
    def __init__(self):
        super(DCVAE, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def generate(self, N):
        x = torch.randn([N, config.PARAM['latent_size']], device=config.PARAM['device'])
        x = self.model['decoder_latent'](x)
        x = x.view(x.size(0), *config.PARAM['encode_shape'])
        generated = self.model['decoder'](x)
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['img']
        x = self.model['encoder'](x)
        x = x.view(x.size(0), -1)
        latent_encoded = self.model['encoder_latent'](x)
        output['mu'], output['logvar'] = torch.chunk(latent_encoded, 2, dim=1)
        if self.training:
            x = reparameterize(output['mu'], output['logvar'])
        else:
            x = output['mu']
        x = self.model['decoder_latent'](x)
        x = x.view(x.size(0), *config.PARAM['encode_shape'])
        decoded = self.model['decoder'](x)
        output['img'] = decoded
        output['loss'] = loss(input, output)
        return output


class DCCVAE(nn.Module):
    def __init__(self):
        super(DCCVAE, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def generate(self, C):
        x = torch.randn([C.size(0), config.PARAM['latent_size']], device=config.PARAM['device'])
        onehot = idx2onehot(C)
        decoder_embedding = self.model['decoder_embedding'](onehot)
        x = torch.cat((x, decoder_embedding), dim=1)
        x = self.model['decoder_latent'](x)
        x = x.view(x.size(0), *config.PARAM['encode_shape'])
        generated = self.model['decoder'](x)
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['img']
        onehot = idx2onehot(input[config.PARAM['subset']])
        encoder_embedding = self.model['encoder_embedding'](onehot)
        encoder_embedding = onehot.view([*encoder_embedding.size(), 1, 1]).expand(
            [*encoder_embedding.size(), *x.size()[2:]])
        x = torch.cat((x, encoder_embedding), dim=1)
        x = self.model['encoder'](x)
        x = x.view(x.size(0), -1)
        latent_encoded = self.model['encoder_latent'](x)
        output['mu'], output['logvar'] = torch.chunk(latent_encoded, 2, dim=1)
        if self.training:
            x = reparameterize(output['mu'], output['logvar'])
        else:
            x = output['mu']
        decoder_embedding = self.model['decoder_embedding'](onehot)
        x = torch.cat((x, decoder_embedding), dim=1)
        x = self.model['decoder_latent'](x)
        x = x.view(x.size(0), *config.PARAM['encode_shape'])
        decoded = self.model['decoder'](x)
        output['img'] = decoded
        output['loss'] = loss(input, output)
        return output


def dcvae():
    normalization = 'bn'
    activation = 'relu'
    img_shape = config.PARAM['img_shape']
    latent_size = config.PARAM['latent_size']
    hidden_size = config.PARAM['hidden_size']
    depth = config.PARAM['depth']
    encode_shape = config.PARAM['encode_shape']
    config.PARAM['model'] = {}
    # Encoder
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': img_shape[0], 'output_size': hidden_size,
         'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': True, 'normalization': normalization,
         'activation': activation})
    for i in range(depth):
        config.PARAM['model']['encoder'].append(
            {'cell': 'Conv2dCell', 'input_size': hidden_size * (2 ** i), 'output_size': hidden_size * (2 ** (i + 1)),
             'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
             'activation': activation})
    config.PARAM['model']['encoder'].append(
        {'cell': 'ResConv2dCell', 'input_size': hidden_size * (2 ** depth), 'output_size': hidden_size * (2 ** depth),
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    config.PARAM['model']['encoder'].append(
        {'cell': 'ResConv2dCell', 'input_size': hidden_size * (2 ** depth), 'output_size': hidden_size * (2 ** depth),
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    # Latent
    config.PARAM['model']['encoder_latent'] = {
        'cell': 'LinearCell', 'input_size': np.prod(encode_shape), 'output_size': 2 * latent_size,
        'bias': True, 'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['decoder_latent'] = {
        'cell': 'LinearCell', 'input_size': latent_size, 'output_size': np.prod(encode_shape),
        'bias': True, 'normalization': 'none', 'activation': activation}
    # Decoder
    config.PARAM['model']['decoder'] = []
    config.PARAM['model']['decoder'].append(
        {'cell': 'ResConv2dCell', 'input_size': hidden_size * (2 ** depth), 'output_size': hidden_size * (2 ** depth),
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    config.PARAM['model']['decoder'].append(
        {'cell': 'ResConv2dCell', 'input_size': hidden_size * (2 ** depth), 'output_size': hidden_size * (2 ** depth),
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    for i in range(depth):
        config.PARAM['model']['decoder'].append(
            {'cell': 'ConvTranspose2dCell', 'input_size': hidden_size * (2 ** (depth - i)),
             'output_size': hidden_size * (2 ** (depth - i - 1)),
             'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
             'activation': activation})
    config.PARAM['model']['decoder'].append(
        {'cell': 'Conv2dCell', 'input_size': hidden_size, 'output_size': img_shape[0],
         'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': True, 'normalization': 'none',
         'activation': 'sigmoid'})
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = DCVAE()
    return model


def dccvae():
    normalization = 'bn'
    activation = 'relu'
    img_shape = config.PARAM['img_shape']
    encode_shape = config.PARAM['encode_shape']
    latent_size = config.PARAM['latent_size']
    hidden_size = config.PARAM['hidden_size']
    depth = config.PARAM['depth']
    classes_size = int(config.PARAM['classes_size'])
    config.PARAM['model'] = {}
    # Embedding
    config.PARAM['model']['encoder_embedding'] = {
        'cell': 'LinearCell', 'input_size': classes_size, 'output_size': classes_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['decoder_embedding'] = {
        'cell': 'LinearCell', 'input_size': classes_size, 'output_size': classes_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    # Encoder
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': img_shape[0] + classes_size, 'output_size': hidden_size,
         'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': True, 'normalization': normalization,
         'activation': activation})
    for i in range(depth):
        config.PARAM['model']['encoder'].append(
            {'cell': 'Conv2dCell', 'input_size': hidden_size * (2 ** i), 'output_size': hidden_size * (2 ** (i + 1)),
             'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
             'activation': activation})
    config.PARAM['model']['encoder'].append(
        {'cell': 'ResConv2dCell', 'input_size': hidden_size * (2 ** depth), 'output_size': hidden_size * (2 ** depth),
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    config.PARAM['model']['encoder'].append(
        {'cell': 'ResConv2dCell', 'input_size': hidden_size * (2 ** depth), 'output_size': hidden_size * (2 ** depth),
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    # Latent
    config.PARAM['model']['encoder_latent'] = {
        'cell': 'LinearCell', 'input_size': np.prod(encode_shape), 'output_size': 2 * latent_size,
        'bias': True, 'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['decoder_latent'] = {
        'cell': 'LinearCell', 'input_size': latent_size + classes_size, 'output_size': np.prod(encode_shape),
        'bias': True, 'normalization': 'none', 'activation': activation}
    # Decoder
    config.PARAM['model']['decoder'] = []
    config.PARAM['model']['decoder'].append(
        {'cell': 'ResConv2dCell', 'input_size': hidden_size * (2 ** depth), 'output_size': hidden_size * (2 ** depth),
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    config.PARAM['model']['decoder'].append(
        {'cell': 'ResConv2dCell', 'input_size': hidden_size * (2 ** depth), 'output_size': hidden_size * (2 ** depth),
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    for i in range(depth):
        config.PARAM['model']['decoder'].append(
            {'cell': 'ConvTranspose2dCell', 'input_size': hidden_size * (2 ** (depth - i)),
             'output_size': hidden_size * (2 ** (depth - i - 1)),
             'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
             'activation': activation})
    config.PARAM['model']['decoder'].append(
        {'cell': 'Conv2dCell', 'input_size': hidden_size, 'output_size': img_shape[0],
         'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': True, 'normalization': 'none',
         'activation': 'sigmoid'})
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = DCCVAE()
    return model