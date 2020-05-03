import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_model


class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def encode(self, input):
        x = input['img']
        x = self.model['encoder'](x)
        _, _, code = self.model['quantizer'](x)
        return code

    def decode(self, code):
        x = self.model['quantizer'].embedding_code(code).permute(0, 3, 1, 2).contiguous()
        generated = self.model['decoder'](x)
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['img']
        x = self.model['encoder'](x)
        x, vq_loss, output['idx'] = self.model['quantizer'](x)
        decoded = self.model['decoder'](x)
        output['img'] = decoded
        output['loss'] = F.mse_loss(decoded, input['img']) + config.PARAM['vq_commit'] * vq_loss
        return output


def vqvae():
    normalization = 'bn'
    activation = 'relu'
    img_shape = config.PARAM['img_shape']
    hidden_size = config.PARAM['hidden_size']
    quantizer_embedding_size = config.PARAM['quantizer_embedding_size']
    num_embedding = config.PARAM['num_embedding']
    config.PARAM['model'] = {}
    # Encoder
    input_size = img_shape[0]
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
    output_size = hidden_size
    for i in range(2):
        config.PARAM['model']['encoder'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
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
    input_size = quantizer_embedding_size
    output_size = hidden_size
    config.PARAM['model']['decoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size
    output_size = hidden_size
    for i in range(2):
        config.PARAM['model']['decoder'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
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
    model = VQVAE()
    return model