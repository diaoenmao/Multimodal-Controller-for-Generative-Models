import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_model


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.model = nn.ModuleDict({})
        self.model['encoder'] = nn.ModuleList([encoder] * config.PARAM['split_encoder'])
        if config.PARAM['split_mode_model'] == 0:
            self.model['decoder'] = decoder
        else:
            self.model['decoder'] = nn.ModuleList([decoder] * config.PARAM['split_encoder'])

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32), 'img': None,
                  'code': None}
        if self.training:
            actived = input['label'].unique().tolist()
            mask = []
            for i in range(len(actived)):
                mask.append(input['label'] == actived[i])
            x = input['img']
            encoded = None
            for i in range(len(actived)):
                if i > len(self.model['encoder']) - 1:
                    continue
                x_i = x[mask[i]]
                encoded_i = self.model['encoder'][actived[i]](x_i)
                if encoded is None:
                    encoded_shape = [x.size(0), *encoded_i.shape[1:]]
                    encoded = x.new_zeros(encoded_shape)
                encoded[mask[i]] = encoded_i
            decoded = x.new_zeros(x.size())
            if config.PARAM['split_mode_model'] == 0:
                decoded = self.model['decoder'](encoded)
            else:
                for i in range(len(actived)):
                    if i > len(self.model['decoder']) - 1:
                        continue
                    encoded_i = encoded[mask[i]]
                    decoded_i = self.model['decoder'][actived[i]](encoded_i)
                    decoded[mask[i]] = decoded_i
            output['img'] = decoded
            output['loss'] = F.mse_loss(output['img'], input['img'])
        else:
            x = input['img']
            encoded = []
            for i in range(len(self.model['encoder'])):
                x_i = x
                encoded_i = self.model['encoder'][i](x_i)
                if encoded is None:
                    encoded_shape = [x.size(0), *encoded_i.shape[1:]]
                    encoded = x.new_zeros(encoded_shape)
                encoded.append(encoded_i)
            if config.PARAM['split_mode_model'] == 0:
                encoded = torch.cat(encoded, dim=0)
                decoded = self.model['decoder'](encoded)
                decoded = list(torch.chunk(decoded, len(self.model['encoder']), dim=0))
            else:
                decoded = []
                for i in range(len(self.model['decoder'])):
                    encoded_i = encoded[i]
                    decoded_i = self.model['decoder'][i](encoded_i)
                    decoded.append(decoded_i)
            output['img'] = decoded
            for i in range(len(self.model['encoder'])):
                output['loss'] += F.mse_loss(output['img'][i], input['img'])
            output['loss'] /= len(self.model['encoder'])
        return output


def fae():
    num_channel = config.PARAM['num_channel']
    num_hidden_encoder = config.PARAM['num_hidden_encoder']
    num_hidden_decoder = config.PARAM['num_hidden_decoder']
    scale_factor = config.PARAM['scale_factor']
    depth = config.PARAM['depth']

    config.PARAM['model'] = {}
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'ConvCell', 'input_size': num_channel, 'output_size': num_hidden_encoder,
         'kernel_size': 1, 'stride': 1, 'padding': 0})
    for i in range(depth):
        config.PARAM['model']['encoder'].append(
            {'cell': 'ResConvCell', 'input_size': num_hidden_encoder * ((2 * scale_factor) ** i),
             'output_size': num_hidden_encoder * ((2 * scale_factor) ** i), 'kernel_size': 3, 'stride': 1,
             'padding': 1})
        config.PARAM['model']['encoder'].append({'cell': 'ShuffleCell', 'mode': 'down', 'scale_factor': scale_factor})

    config.PARAM['model']['encoder'].append(
        {'cell': 'ConvCell', 'input_size': num_hidden_encoder * ((2 * scale_factor) ** depth),
         'output_size': num_hidden_encoder, 'kernel_size': 1, 'stride': 1, 'padding': 0})

    config.PARAM['model']['decoder'] = []
    config.PARAM['model']['decoder'].append(
        {'cell': 'ConvCell', 'input_size': num_hidden_encoder,
         'output_size': num_hidden_decoder * ((2 * scale_factor) ** depth),
         'kernel_size': 1, 'stride': 1, 'padding': 0})
    for i in range(depth):
        config.PARAM['model']['decoder'].append({'cell': 'ShuffleCell', 'mode': 'up', 'scale_factor': scale_factor})
        config.PARAM['model']['decoder'].append(
            {'cell': 'ResConvCell', 'input_size': num_hidden_decoder * ((2 * scale_factor) ** (depth - i - 1)),
             'output_size': num_hidden_decoder * ((2 * scale_factor) ** (depth - i - 1)),
             'kernel_size': 3, 'stride': 1, 'padding': 1})
    config.PARAM['model']['decoder'].append(
        {'cell': 'ConvCell', 'input_size': num_hidden_decoder, 'output_size': num_channel, 'kernel_size': 1,
         'stride': 1, 'padding': 0, 'activation': 'none'})

    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    encoder = make_model(config.PARAM['model']['encoder'])
    decoder = make_model(config.PARAM['model']['decoder'])
    model = Model(encoder, decoder)
    return model