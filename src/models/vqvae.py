import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_model


class DCCVQVAE(nn.Module):
    def __init__(self):
        super(DCCVQVAE, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def encode(self, input):
        x = input['img']
        onehot = F.one_hot(input['label'], config.PARAM['classes_size']).float()
        encoder_embedding = self.model['encoder_embedding'](onehot)
        encoder_embedding = encoder_embedding.view([*encoder_embedding.size(), 1, 1]).expand(
            [*encoder_embedding.size(), *x.size()[2:]])
        x = torch.cat((x, encoder_embedding), dim=1)
        x = self.model['encoder'](x)
        _, _, code = self.model['quantizer'](x)
        return code

    def decode(self, code, label):
        onehot = F.one_hot(label, config.PARAM['classes_size']).float()
        x = self.model['quantizer'].embedding_code(code).permute(0, 3, 1, 2).contiguous()
        decoder_embedding = self.model['decoder_embedding'](onehot)
        decoder_embedding = decoder_embedding.view([*decoder_embedding.size(), 1, 1]).expand(
            [*decoder_embedding.size(), *x.size()[2:]])
        x = torch.cat((x, decoder_embedding), dim=1)
        generated = self.model['decoder'](x)
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['img']
        onehot = F.one_hot(input['label'], config.PARAM['classes_size']).float()
        encoder_embedding = self.model['encoder_embedding'](onehot)
        encoder_embedding = encoder_embedding.view([*encoder_embedding.size(), 1, 1]).expand(
            [*encoder_embedding.size(), *x.size()[2:]])
        x = torch.cat((x, encoder_embedding), dim=1)
        x = self.model['encoder'](x)
        x, vq_loss, output['idx'] = self.model['quantizer'](x)
        decoder_embedding = self.model['decoder_embedding'](onehot)
        decoder_embedding = decoder_embedding.view([*decoder_embedding.size(), 1, 1]).expand(
            [*decoder_embedding.size(), *x.size()[2:]])
        x = torch.cat((x, decoder_embedding), dim=1)
        decoded = self.model['decoder'](x)
        output['img'] = decoded
        output['loss'] = F.mse_loss(decoded, input['img']) + config.PARAM['vq_commit'] * vq_loss
        return output


class DCMCVQVAE(nn.Module):
    def __init__(self):
        super(DCMCVQVAE, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def encode(self, input):
        x = input['img']
        config.PARAM['indicator'] = F.one_hot(input['label'], config.PARAM['classes_size']).float()
        x = self.model['encoder'](x)
        _, _, code = self.model['quantizer'](x)
        return code

    def decode(self, code, label):
        config.PARAM['indicator'] = F.one_hot(label, config.PARAM['classes_size']).float()
        x = self.model['quantizer'].embedding_code(code).permute(0, 3, 1, 2).contiguous()
        generated = self.model['decoder'](x)
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['img']
        config.PARAM['indicator'] = F.one_hot(input['label'], config.PARAM['classes_size']).float()
        x = self.model['encoder'](x)
        x, vq_loss, output['idx'] = self.model['quantizer'](x)
        decoded = self.model['decoder'](x)
        output['img'] = decoded
        output['loss'] = F.mse_loss(decoded, input['img']) + config.PARAM['vq_commit'] * vq_loss
        return output


def dccvqvae():
    normalization = 'bn'
    activation = 'relu'
    img_shape = config.PARAM['img_shape']
    num_mode = config.PARAM['classes_size']
    hidden_size = config.PARAM['hidden_size']
    conditional_embedding_size = config.PARAM['conditional_embedding_size']
    quantizer_embedding_size = config.PARAM['quantizer_embedding_size']
    num_embedding = config.PARAM['num_embedding']
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
    output_size = hidden_size
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size
    output_size = hidden_size
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size
    res_size = hidden_size
    output_size = hidden_size
    for i in range(2):
        config.PARAM['model']['encoder'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'res_size': res_size,
             'normalization': normalization, 'activation': activation})
    input_size = hidden_size
    output_size = quantizer_embedding_size
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': 'none',
         'activation': 'none'})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    # Quantizer
    config.PARAM['model']['quantizer'] = {
        'cell': 'VectorQuantization', 'embedding_dim': quantizer_embedding_size, 'num_embedding': num_embedding}
    # Decoder
    config.PARAM['model']['decoder'] = []
    input_size = quantizer_embedding_size + conditional_embedding_size
    output_size = hidden_size
    config.PARAM['model']['decoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size
    res_size = hidden_size
    output_size = hidden_size
    for i in range(2):
        config.PARAM['model']['decoder'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'res_size': res_size,
             'normalization': normalization, 'activation': activation})
    input_size = hidden_size
    output_size = hidden_size
    config.PARAM['model']['decoder'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size
    output_size = img_shape[0]
    config.PARAM['model']['decoder'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': 'none',
         'activation': 'tanh'})
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = DCCVQVAE()
    return model


def dcmcvqvae():
    normalization = 'bn'
    activation = 'relu'
    img_shape = config.PARAM['img_shape']
    num_mode = config.PARAM['classes_size']
    controller_rate = config.PARAM['controller_rate']
    hidden_size = config.PARAM['hidden_size']
    quantizer_embedding_size = config.PARAM['quantizer_embedding_size']
    num_embedding = config.PARAM['num_embedding']
    config.PARAM['model'] = {}
    # Encoder
    input_size = img_shape[0]
    output_size = hidden_size
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'MCConv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = hidden_size
    output_size = hidden_size
    config.PARAM['model']['encoder'].append(
        {'cell': 'MCConv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = hidden_size
    res_size = hidden_size
    output_size = hidden_size
    for i in range(2):
        config.PARAM['model']['encoder'].append(
            {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'res_size': res_size,
             'normalization': normalization, 'activation': activation, 'num_mode': num_mode,
             'controller_rate': controller_rate})
    input_size = hidden_size
    output_size = quantizer_embedding_size
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': 'none',
         'activation': 'none', 'num_mode': num_mode, 'controller_rate': controller_rate})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    # Quantizer
    config.PARAM['model']['quantizer'] = {
        'cell': 'VectorQuantization', 'embedding_dim': quantizer_embedding_size, 'num_embedding': num_embedding}
    # Decoder
    config.PARAM['model']['decoder'] = []
    input_size = quantizer_embedding_size
    output_size = hidden_size
    config.PARAM['model']['decoder'].append(
        {'cell': 'MCConv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = hidden_size
    res_size = hidden_size
    output_size = hidden_size
    for i in range(2):
        config.PARAM['model']['decoder'].append(
            {'cell': 'MCResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'res_size': res_size,
             'normalization': normalization, 'activation': activation, 'num_mode': num_mode,
             'controller_rate': controller_rate})
    input_size = hidden_size
    output_size = hidden_size
    config.PARAM['model']['decoder'].append(
        {'cell': 'MCConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation, 'num_mode': num_mode, 'controller_rate': controller_rate})
    input_size = hidden_size
    output_size = img_shape[0]
    config.PARAM['model']['decoder'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': 'none',
         'activation': 'tanh'})
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = DCMCVQVAE()
    return model