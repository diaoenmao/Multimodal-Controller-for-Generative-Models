import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_model


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32), 'img': None,
                  'code': None}
        x = input['img']
        encoded = self.model['encoder'](x)
        encoded = encoded.unsqueeze(1)
        quantized, encoding, distances, quantization_loss = self.model['quantizer'](encoded)
        quantized = quantized.squeeze(1)
        decoded = self.model['decoder'](quantized)
        output['img'] = decoded
        output['loss'] = F.mse_loss(output['img'], input['img']) / torch.var(input['img']) + quantization_loss
        return output


def cae():
    num_channel = config.PARAM['num_channel']
    num_hidden = config.PARAM['num_hidden']
    scale_factor = config.PARAM['scale_factor']
    depth = config.PARAM['depth']
    num_embedding = config.PARAM['num_embedding']
    embedding_dim = config.PARAM['embedding_dim']
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append({'cell': 'ConvCell', 'input_size': num_channel, 'output_size': num_hidden,
                                             'kernel_size': 1, 'stride': 1, 'padding': 0})
    for i in range(depth):
        config.PARAM['model']['encoder'].append(
            {'cell': 'ResConvCell', 'input_size': num_hidden * ((2 * scale_factor) ** i),
             'output_size': num_hidden * ((2 * scale_factor) ** i), 'kernel_size': 3, 'stride': 1, 'padding': 1})
        config.PARAM['model']['encoder'].append({'cell': 'ShuffleCell', 'mode': 'down', 'scale_factor': scale_factor})

    config.PARAM['model']['encoder'].append(
        {'cell': 'ConvCell', 'input_size': num_hidden * ((2 * scale_factor) ** depth),
         'output_size': embedding_dim, 'kernel_size': 1, 'stride': 1, 'padding': 0})

    config.PARAM['model']['quantizer'] = {'cell': 'QuantizationCell', 'num_embedding': num_embedding,
                                          'embedding_dim': embedding_dim}
    config.PARAM['model']['decoder'] = []
    config.PARAM['model']['decoder'].append(
        {'cell': 'ConvCell', 'input_size': embedding_dim, 'output_size': num_hidden * ((2 * scale_factor) ** depth),
         'kernel_size': 1, 'stride': 1, 'padding': 0})
    for i in range(depth):
        config.PARAM['model']['decoder'].append({'cell': 'ShuffleCell', 'mode': 'up', 'scale_factor': scale_factor})
        config.PARAM['model']['decoder'].append(
            {'cell': 'ResConvCell', 'input_size': num_hidden * ((2 * scale_factor) ** (depth - i - 1)),
             'output_size': num_hidden * ((2 * scale_factor) ** (depth - i - 1)),
             'kernel_size': 3, 'stride': 1, 'padding': 1})
    config.PARAM['model']['decoder'].append(
        {'cell': 'ConvCell', 'input_size': num_hidden, 'output_size': num_channel, 'kernel_size': 1,
         'stride': 1, 'padding': 0, 'activation': 'none'})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = Model()
    return model


def caes():
    num_channel = config.PARAM['num_channel']
    num_hidden = config.PARAM['num_hidden']
    scale_factor = config.PARAM['scale_factor']
    depth = config.PARAM['depth']
    num_embedding = config.PARAM['num_embedding']
    embedding_dim = config.PARAM['embedding_dim']
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append({'cell': 'ConvCell', 'input_size': num_channel, 'output_size': num_hidden,
                                             'kernel_size': 1, 'stride': 1, 'padding': 0})
    config.PARAM['model']['encoder'].append(
        {'cell': 'ShuffleCell', 'mode': 'down', 'scale_factor': scale_factor ** depth})

    config.PARAM['model']['encoder'].append(
        {'cell': 'ConvCell', 'input_size': num_hidden * ((2 * scale_factor) ** depth),
         'output_size': num_hidden, 'kernel_size': 3, 'stride': 1, 'padding': 1})

    config.PARAM['model']['quantizer'] = {'cell': 'QuantizationCell', 'num_embedding': num_embedding,
                                          'embedding_dim': embedding_dim}
    config.PARAM['model']['decoder'] = []
    config.PARAM['model']['decoder'].append(
        {'cell': 'ConvCell', 'input_size': num_hidden, 'output_size': num_hidden * ((2 * scale_factor) ** depth),
         'kernel_size': 3, 'stride': 1, 'padding': 1})
    config.PARAM['model']['decoder'].append(
        {'cell': 'ShuffleCell', 'mode': 'up', 'scale_factor': scale_factor ** depth})
    for i in range(depth):
        config.PARAM['model']['decoder'].append(
            {'cell': 'ResConvCell', 'input_size': num_hidden,
             'output_size': num_hidden, 'kernel_size': 3, 'stride': 1, 'padding': 1})
    config.PARAM['model']['decoder'].append(
        {'cell': 'ConvCell', 'input_size': num_hidden, 'output_size': num_channel, 'kernel_size': 1,
         'stride': 1, 'padding': 0, 'activation': 'none'})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = Model()
    return model