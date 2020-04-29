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


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def generate(self, x, C):
        onehot = F.one_hot(C, config.PARAM['classes_size']).float()
        decoder_embedding = self.model['decoder_embedding'](onehot)
        x = torch.cat((x, decoder_embedding), dim=1)
        generated = self.model['decoder'](x)
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        input['img'] = (input['img'] + 1) / 2
        x = input['img']
        onehot = F.one_hot(input['label'], config.PARAM['classes_size']).float()
        encoder_embedding = self.model['encoder_embedding'](onehot)
        encoder_embedding = encoder_embedding.view([*encoder_embedding.size(), 1, 1]).expand(
            [*encoder_embedding.size(), *x.size()[2:]])
        x = torch.cat((x, encoder_embedding), dim=1)
        x = self.model['encoder'](x)
        output['mu'], output['logvar'] = self.model['mu'](x), self.model['logvar'](x)
        x = reparameterize(output['mu'], output['logvar']) if self.training else output['mu']
        decoder_embedding = self.model['decoder_embedding'](onehot)
        x = torch.cat((x, decoder_embedding), dim=1)
        decoded = self.model['decoder'](x)
        output['img'] = decoded
        output['loss'] = loss(input, output)
        return output


class MCVAE(nn.Module):
    def __init__(self):
        super(MCVAE, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def generate(self, x, C):
        config.PARAM['indicator'] = F.one_hot(C, config.PARAM['classes_size']).float()
        generated = self.model['decoder'](x)
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        input['img'] = (input['img'] + 1) / 2
        x = input['img']
        config.PARAM['indicator'] = F.one_hot(input['label'], config.PARAM['classes_size']).float()
        x = self.model['encoder'](x)
        output['mu'] = self.model['mu'](x)
        output['logvar'] = self.model['logvar'](x)
        output['mu'], output['logvar'] = self.model['mu'](x), self.model['logvar'](x)
        x = reparameterize(output['mu'], output['logvar']) if self.training else output['mu']
        decoded = self.model['decoder'](x)
        output['img'] = decoded
        output['loss'] = loss(input, output)
        return output


def cvae():
    normalization = 'bn'
    activation = 'relu'
    img_shape = config.PARAM['img_shape']
    num_mode = config.PARAM['classes_size']
    conditional_embedding_size = config.PARAM['conditional_embedding_size']
    hidden_size = config.PARAM['hidden_size']
    latent_size = config.PARAM['latent_size']
    encode_shape = config.PARAM['encode_shape']
    config.PARAM['model'] = {}
    # Embedding
    config.PARAM['model']['encoder_embedding'] = {
        'cell': 'LinearCell', 'input_size': num_mode, 'output_size': conditional_embedding_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['decoder_embedding'] = {
        'cell': 'LinearCell', 'input_size': num_mode, 'output_size': conditional_embedding_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    # Encoder
    input_size = img_shape[0] + conditional_embedding_size
    output_size = hidden_size[0]
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size[0]
    output_size = hidden_size[1]
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size[1]
    output_size = hidden_size[2]
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size[2]
    res_size = hidden_size[2]
    output_size = hidden_size[2]
    for i in range(2):
        config.PARAM['model']['encoder'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'res_size': res_size,
             'normalization': normalization, 'activation': activation})
    config.PARAM['model']['encoder'].append({'cell': 'ResizeCell', 'resize': [-1]})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    # Latent
    config.PARAM['model']['mu'] = {
        'cell': 'LinearCell', 'input_size': np.prod(encode_shape), 'output_size': latent_size,
        'bias': True, 'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['logvar'] = {
        'cell': 'LinearCell', 'input_size': np.prod(encode_shape), 'output_size': latent_size,
        'bias': True, 'normalization': 'none', 'activation': 'none'}
    # Decoder
    config.PARAM['model']['decoder'] = []
    input_size = latent_size + conditional_embedding_size
    output_size = np.prod(encode_shape)
    config.PARAM['model']['decoder'].append(
        {'cell': 'LinearCell', 'input_size': input_size, 'output_size': output_size,
         'bias': True, 'normalization': normalization, 'activation': activation})
    config.PARAM['model']['decoder'].append(
        {'cell': 'ResizeCell', 'resize': encode_shape})
    input_size = hidden_size[2]
    res_size = hidden_size[2]
    output_size = hidden_size[2]
    for i in range(2):
        config.PARAM['model']['decoder'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'res_size': res_size,
             'normalization': normalization, 'activation': activation})
    input_size = hidden_size[2]
    output_size = hidden_size[1]
    config.PARAM['model']['decoder'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size[1]
    output_size = hidden_size[0]
    config.PARAM['model']['decoder'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size[0]
    output_size = img_shape[0]
    config.PARAM['model']['decoder'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': 'none',
         'activation': 'sigmoid'})
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = CVAE()
    return model


def mcvae():
    normalization = 'bn'
    activation = 'relu'
    img_shape = config.PARAM['img_shape']
    num_mode = config.PARAM['classes_size']
    controller_rate = config.PARAM['controller_rate']
    hidden_size = config.PARAM['hidden_size']
    latent_size = config.PARAM['latent_size']
    encode_shape = config.PARAM['encode_shape']
    config.PARAM['model'] = {}
    # Encoder
    input_size = img_shape[0]
    output_size = hidden_size[0]
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'MCConv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = hidden_size[0]
    output_size = hidden_size[1]
    config.PARAM['model']['encoder'].append(
        {'cell': 'MCConv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = hidden_size[1]
    output_size = hidden_size[2]
    config.PARAM['model']['encoder'].append(
        {'cell': 'MCConv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = hidden_size[2]
    res_size = hidden_size[2]
    output_size = hidden_size[2]
    for i in range(2):
        config.PARAM['model']['encoder'].append(
            {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'res_size': res_size,
             'normalization': normalization, 'activation': activation, 'num_mode': num_mode,
             'controller_rate': controller_rate})
    config.PARAM['model']['encoder'].append({'cell': 'ResizeCell', 'resize': [-1]})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    # Latent
    config.PARAM['model']['mu'] = {
        'cell': 'LinearCell', 'input_size': np.prod(encode_shape), 'output_size': latent_size,
        'bias': True, 'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['logvar'] = {
        'cell': 'LinearCell', 'input_size': np.prod(encode_shape), 'output_size': latent_size,
        'bias': True, 'normalization': 'none', 'activation': 'none'}
    # Decoder
    config.PARAM['model']['decoder'] = []
    input_size = latent_size
    output_size = np.prod(encode_shape)
    config.PARAM['model']['decoder'].append(
        {'cell': 'LinearCell', 'input_size': input_size, 'output_size': output_size,
         'bias': True, 'normalization': normalization, 'activation': activation})
    config.PARAM['model']['decoder'].append(
        {'cell': 'ResizeCell', 'resize': encode_shape})
    config.PARAM['model']['decoder'].append(
        {'cell': 'MultimodalController', 'input_size': hidden_size[2], 'num_mode': num_mode,
         'controller_rate': controller_rate})
    input_size = hidden_size[2]
    res_size = hidden_size[2]
    output_size = hidden_size[2]
    for i in range(2):
        config.PARAM['model']['decoder'].append(
            {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'res_size': res_size,
             'normalization': normalization, 'activation': activation, 'num_mode': num_mode,
             'controller_rate': controller_rate})
    input_size = hidden_size[2]
    output_size = hidden_size[1]
    config.PARAM['model']['decoder'].append(
        {'cell': 'MCConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = hidden_size[1]
    output_size = hidden_size[0]
    config.PARAM['model']['decoder'].append(
        {'cell': 'MCConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = hidden_size[0]
    output_size = img_shape[0]
    config.PARAM['model']['decoder'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': 'none',
         'activation': 'sigmoid'})
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = MCVAE()
    return model