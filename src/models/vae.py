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
    CE = F.binary_cross_entropy_with_logits(output['img'], input['img'], reduction='sum')
    KLD = -0.5 * torch.sum(1 + output['logvar'] - output['mu'].pow(2) - output['logvar'].exp())
    return CE + KLD


def idx2onehot(idx, classes_size):
    idx = idx.view(idx.size(0), 1)
    onehot = idx.new_zeros(idx.size(0), classes_size).float()
    onehot.scatter_(1, idx, 1)
    return onehot


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def generate(self, N):
        x = torch.randn([N, config.PARAM['latent_size']]).to(config.PARAM['device'])
        generated = self.model['decoder'](x)
        generated = generated.view(generated.size(0), *config.PARAM['img_shape'])
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['img']
        x = x.view(x.size(0), -1)
        encoded = self.model['encoder'](x)
        output['mu'], output['logvar'] = torch.chunk(encoded, 2, dim=1)
        if self.training:
            x = reparameterize(output['mu'], output['logvar'])
        else:
            x = output['mu']
        decoded = self.model['decoder'](x)
        output['img'] = decoded.view(decoded.size(0), *config.PARAM['img_shape'])
        output['loss'] = loss(input, output)
        return output


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def generate(self, c):
        N = c.size(0)
        onehot = idx2onehot(c, config.PARAM['classes_size'])
        x = torch.randn([N, config.PARAM['latent_size']]).to(config.PARAM['device'])
        x = torch.cat((x, onehot), dim=-1)
        generated = self.model['decoder'](x)
        generated = generated.view(generated.size(0), *config.PARAM['img_shape'])
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x, c = input['img'], input['label']
        x = x.view(x.size(0), -1)
        onehot = idx2onehot(c, config.PARAM['classes_size'])
        x = torch.cat((x, onehot), dim=-1)
        encoded = self.model['encoder'](x)
        output['mu'], output['logvar'] = torch.chunk(encoded, 2, dim=1)
        if self.training:
            x = reparameterize(output['mu'], output['logvar'])
        else:
            x = output['mu']
        x = torch.cat((x, onehot), dim=-1)
        decoded = self.model['decoder'](x)
        output['img'] = decoded.view(decoded.size(0), *config.PARAM['img_shape'])
        output['loss'] = loss(input, output)
        return output


def vae():
    normalization = config.PARAM['normalization']
    activation = config.PARAM['activation']
    hidden_size = int(config.PARAM['hidden_size'])
    latent_size = int(config.PARAM['latent_size'])
    num_layers = int(config.PARAM['num_layers'])
    feature_size = np.prod(config.PARAM['img_shape'])
    config.PARAM['model'] = {}
    # Encoder
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'LinearCell', 'input_size': feature_size, 'output_size': hidden_size, 'bias': False,
         'normalization': normalization, 'activation': activation})
    for i in range(num_layers - 2):
        config.PARAM['model']['encoder'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': hidden_size, 'bias': False,
             'normalization': normalization, 'activation': activation})
    config.PARAM['model']['encoder'].append(
        {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': 2 * latent_size, 'bias': False,
         'normalization': 'none', 'activation': 'none'})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    # Decoder
    config.PARAM['model']['decoder'] = []
    config.PARAM['model']['decoder'].append(
        {'cell': 'LinearCell', 'input_size': latent_size, 'output_size': hidden_size, 'bias': False,
         'normalization': normalization, 'activation': activation})
    for i in range(num_layers - 2):
        config.PARAM['model']['decoder'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': hidden_size, 'bias': False,
             'normalization': normalization, 'activation': activation})
    config.PARAM['model']['decoder'].append(
        {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': feature_size, 'bias': False,
         'normalization': 'none', 'activation': 'none'})
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = VAE()
    return model


def cvae():
    normalization = config.PARAM['normalization']
    activation = config.PARAM['activation']
    hidden_size = int(config.PARAM['hidden_size'])
    latent_size = int(config.PARAM['latent_size'])
    num_layers = int(config.PARAM['num_layers'])
    feature_size = np.prod(config.PARAM['img_shape'])
    classes_size = config.PARAM['classes_size']
    config.PARAM['model'] = {}
    # Encoder
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'LinearCell', 'input_size': feature_size + classes_size, 'output_size': hidden_size, 'bias': False,
         'normalization': normalization, 'activation': activation})
    for i in range(num_layers - 2):
        config.PARAM['model']['encoder'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': hidden_size, 'bias': False,
             'normalization': normalization, 'activation': activation})
    config.PARAM['model']['encoder'].append(
        {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': 2 * latent_size, 'bias': False,
         'normalization': 'none', 'activation': 'none'})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    # Decoder
    config.PARAM['model']['decoder'] = []
    config.PARAM['model']['decoder'].append(
        {'cell': 'LinearCell', 'input_size': latent_size + classes_size, 'output_size': hidden_size, 'bias': False,
         'normalization': normalization, 'activation': activation})
    for i in range(num_layers - 2):
        config.PARAM['model']['decoder'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': hidden_size, 'bias': False,
             'normalization': normalization, 'activation': activation})
    config.PARAM['model']['decoder'].append(
        {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': feature_size, 'bias': False,
         'normalization': 'none', 'activation': 'none'})
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = CVAE()
    return model