import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
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
        self.model = make_model(cfg['model'])

    def generate(self, C, x=None):
        if x is None:
            x = torch.randn([C.size(0), cfg['latent_size']], device=cfg['device'])
        onehot = F.one_hot(C, cfg['classes_size']).float()
        decoder_embedding = self.model['decoder_embedding']((onehot,))[0]
        x = torch.cat((x, decoder_embedding), dim=1)
        generated = self.model['decoder']((x,))[0]
        generated = generated * 2 - 1
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        input['img'] = (input['img'] + 1) / 2
        x = input['img']
        onehot = F.one_hot(input['label'], cfg['classes_size']).float()
        encoder_embedding = self.model['encoder_embedding']((onehot,))[0]
        encoder_embedding = encoder_embedding.view([*encoder_embedding.size(), 1, 1]).expand(
            [*encoder_embedding.size(), *x.size()[2:]])
        x = torch.cat((x, encoder_embedding), dim=1)
        x = self.model['encoder']((x,))
        output['mu'], output['logvar'] = self.model['mu'](x)[0], self.model['logvar'](x)[0]
        x = reparameterize(output['mu'], output['logvar']) if self.training else output['mu']
        decoder_embedding = self.model['decoder_embedding']((onehot,))[0]
        x = torch.cat((x, decoder_embedding), dim=1)
        decoded = self.model['decoder']((x,))[0]
        output['img'] = decoded
        output['loss'] = loss(input, output)
        input['img'] = input['img'] * 2 - 1
        output['img'] = output['img'] * 2 - 1
        return output


class MCVAE(nn.Module):
    def __init__(self):
        super(MCVAE, self).__init__()
        self.model = make_model(cfg['model'])

    def generate(self, C, x=None):
        if x is None:
            x = torch.randn([C.size(0), cfg['latent_size']], device=cfg['device'])
        indicator = F.one_hot(C, cfg['classes_size']).float()
        generated = self.model['decoder']((x, indicator))[0]
        generated = generated * 2 - 1
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        input['img'] = (input['img'] + 1) / 2
        x = input['img']
        indicator = F.one_hot(input['label'], cfg['classes_size']).float()
        x = self.model['encoder']((x, indicator))
        output['mu'], output['logvar'] = self.model['mu'](x)[0], self.model['logvar'](x)[0]
        x = reparameterize(output['mu'], output['logvar']) if self.training else output['mu']
        decoded = self.model['decoder']((x, indicator))[0]
        output['img'] = decoded
        output['loss'] = loss(input, output)
        input['img'] = input['img'] * 2 - 1
        output['img'] = output['img'] * 2 - 1
        return output


def cvae():
    normalization = 'bn'
    activation = 'relu'
    img_shape = cfg['img_shape']
    num_mode = cfg['classes_size']
    conditional_embedding_size = cfg['conditional_embedding_size']
    hidden_size = cfg['hidden_size']
    latent_size = cfg['latent_size']
    encode_shape = cfg['encode_shape']
    cfg['model'] = {}
    # Embedding
    cfg['model']['encoder_embedding'] = {
        'cell': 'LinearCell', 'input_size': num_mode, 'output_size': conditional_embedding_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    cfg['model']['decoder_embedding'] = {
        'cell': 'LinearCell', 'input_size': num_mode, 'output_size': conditional_embedding_size,
        'bias': False, 'normalization': 'none', 'activation': 'none'}
    # Encoder
    input_size = img_shape[0] + conditional_embedding_size
    output_size = hidden_size[0]
    cfg['model']['encoder'] = []
    cfg['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size[0]
    output_size = hidden_size[1]
    cfg['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size[1]
    output_size = hidden_size[2]
    cfg['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size[2]
    output_size = hidden_size[2]
    for i in range(2):
        cfg['model']['encoder'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
             'normalization': normalization, 'activation': activation})
    cfg['model']['encoder'].append({'cell': 'ResizeCell', 'resize': [-1]})
    cfg['model']['encoder'] = tuple(cfg['model']['encoder'])
    # Latent
    cfg['model']['mu'] = {
        'cell': 'LinearCell', 'input_size': np.prod(encode_shape), 'output_size': latent_size,
        'bias': True, 'normalization': 'none', 'activation': 'none'}
    cfg['model']['logvar'] = {
        'cell': 'LinearCell', 'input_size': np.prod(encode_shape), 'output_size': latent_size,
        'bias': True, 'normalization': 'none', 'activation': 'none'}
    # Decoder
    cfg['model']['decoder'] = []
    input_size = latent_size + conditional_embedding_size
    output_size = np.prod(encode_shape)
    cfg['model']['decoder'].append(
        {'cell': 'LinearCell', 'input_size': input_size, 'output_size': output_size,
         'bias': True, 'normalization': normalization, 'activation': activation})
    cfg['model']['decoder'].append({'cell': 'ResizeCell', 'resize': encode_shape})
    input_size = hidden_size[2]
    output_size = hidden_size[2]
    for i in range(2):
        cfg['model']['decoder'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
             'normalization': normalization, 'activation': activation})
    input_size = hidden_size[2]
    output_size = hidden_size[1]
    cfg['model']['decoder'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size[1]
    output_size = hidden_size[0]
    cfg['model']['decoder'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size[0]
    output_size = img_shape[0]
    cfg['model']['decoder'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': 'none',
         'activation': 'sigmoid'})
    cfg['model']['decoder'] = tuple(cfg['model']['decoder'])
    model = CVAE()
    return model


def mcvae():
    normalization = 'bn'
    activation = 'relu'
    img_shape = cfg['img_shape']
    num_mode = cfg['classes_size']
    controller_rate = cfg['controller_rate']
    hidden_size = cfg['hidden_size']
    latent_size = cfg['latent_size']
    encode_shape = cfg['encode_shape']
    cfg['model'] = {}
    # Encoder
    input_size = img_shape[0]
    output_size = hidden_size[0]
    cfg['model']['encoder'] = []
    cfg['model']['encoder'].append(
        {'cell': 'MCConv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = hidden_size[0]
    output_size = hidden_size[1]
    cfg['model']['encoder'].append(
        {'cell': 'MCConv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = hidden_size[1]
    output_size = hidden_size[2]
    cfg['model']['encoder'].append(
        {'cell': 'MCConv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = hidden_size[2]
    output_size = hidden_size[2]
    for i in range(2):
        cfg['model']['encoder'].append(
            {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size,
             'hidden_size': output_size, 'normalization': normalization, 'activation': activation, 'num_mode': num_mode,
             'controller_rate': controller_rate})
    cfg['model']['encoder'].append({'cell': 'ResizeCell', 'resize': [-1]})
    cfg['model']['encoder'] = tuple(cfg['model']['encoder'])
    # Latent
    cfg['model']['mu'] = {
        'cell': 'LinearCell', 'input_size': np.prod(encode_shape), 'output_size': latent_size,
        'bias': True, 'normalization': 'none', 'activation': 'none'}
    cfg['model']['logvar'] = {
        'cell': 'LinearCell', 'input_size': np.prod(encode_shape), 'output_size': latent_size,
        'bias': True, 'normalization': 'none', 'activation': 'none'}
    # Decoder
    cfg['model']['decoder'] = []
    input_size = latent_size
    output_size = np.prod(encode_shape)
    cfg['model']['decoder'].append(
        {'cell': 'MultimodalController', 'input_size': input_size, 'num_mode': num_mode,
         'controller_rate': controller_rate})
    cfg['model']['decoder'].append(
        {'cell': 'LinearCell', 'input_size': input_size, 'output_size': output_size,
         'bias': True, 'normalization': normalization, 'activation': activation})
    cfg['model']['decoder'].append({'cell': 'ResizeCell', 'resize': encode_shape})
    cfg['model']['decoder'].append(
        {'cell': 'MultimodalController', 'input_size': hidden_size[2], 'num_mode': num_mode,
         'controller_rate': controller_rate})
    input_size = hidden_size[2]
    output_size = hidden_size[2]
    for i in range(2):
        cfg['model']['decoder'].append(
            {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size,
             'hidden_size': output_size, 'normalization': normalization, 'activation': activation, 'num_mode': num_mode,
             'controller_rate': controller_rate})
    input_size = hidden_size[2]
    output_size = hidden_size[1]
    cfg['model']['decoder'].append(
        {'cell': 'MCConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = hidden_size[1]
    output_size = hidden_size[0]
    cfg['model']['decoder'].append(
        {'cell': 'MCConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = hidden_size[0]
    output_size = img_shape[0]
    cfg['model']['decoder'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': 'none',
         'activation': 'sigmoid'})
    cfg['model']['decoder'] = tuple(cfg['model']['decoder'])
    model = MCVAE()
    return model