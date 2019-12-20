import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_model


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.ModuleDict({})
        self.model = make_model(config.PARAM['model'])

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32),
                  'img': input['img'].new_zeros(input['img'].size()),
                  'code': None}
        if self.training:
            activated = input['label'].unique().tolist()
            mask = []
            for i in range(len(activated)):
                mask.append(input['label'] == activated[i])
            encoded = []
            decoded = []
            quantization_Loss = 0
            for i in range(len(activated)):
                if activated[i] > len(self.model['encoder']) - 1:
                    continue
                x_i = input['img'][mask[i]]
                encoded_i = self.model['encoder'][activated[i]](x_i)
                encoded_i, _, _, quantization_Loss_i = self.model['quantizer'](encoded_i.unsqueeze(1))
                quantization_Loss += quantization_Loss_i
                encoded_i = encoded_i.squeeze(1)
                encoded.append(encoded_i)
                if config.PARAM['split_mode_model'] in [0, 1]:
                    decoded_i = self.model['decoder'](encoded_i)
                else:
                    decoded_i = self.model['decoder'][activated[i]](encoded_i)
                decoded.append(decoded_i)
                output['img'][mask[i]] = decoded_i
            output['loss'] = F.mse_loss(output['img'], input['img']) + quantization_Loss/len(activated)
        else:
            x = input['img']
            activated = input['activated']
            encoded = self.model['encoder'][activated](x)
            encoded, _, _, quantization_Loss = self.model['quantizer'](encoded.unsqueeze(1))
            encoded = encoded.squeeze(1)
            if config.PARAM['split_mode_model'] == 0:
                decoded = self.model['decoder'](encoded)
            else:
                decoded = self.model['decoder'][activated](encoded)
            output['img'] = decoded
            output['loss'] = F.mse_loss(output['img'], input['img']) + quantization_Loss
        return output


def cvae():
    channel_size = config.PARAM['channel_size']
    encoder_hidden_size = config.PARAM['encoder_hidden_size']
    embedding_size = config.PARAM['embedding_size']
    decoder_hidden_size = config.PARAM['decoder_hidden_size']
    scale_factor = config.PARAM['scale_factor']
    depth = config.PARAM['depth']
    split_encoder = config.PARAM['split_encoder']
    split_mode_model = config.PARAM['split_mode_model']

    config.PARAM['model'] = {}
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'ConvCell', 'input_size': channel_size, 'output_size': encoder_hidden_size,
         'kernel_size': 5, 'stride': 1, 'padding': 2})
    config.PARAM['model']['encoder'].append(
        {'cell': 'ShuffleCell', 'mode': 'down', 'scale_factor': scale_factor ** depth})
    config.PARAM['model']['encoder'].append(
        {'cell': 'ConvCell', 'input_size': encoder_hidden_size * ((2 * scale_factor) ** depth),
         'output_size': embedding_size, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'activation':'none'})
    config.PARAM['model']['encoder'] = [tuple(config.PARAM['model']['encoder']) for _ in range(split_encoder)]
    config.PARAM['model']['quantizer'] = {'cell': 'QuantizationCell', 'num_embedding': config.PARAM['num_embedding']}
    if split_mode_model == 0:
        config.PARAM['model']['decoder'] = []
        config.PARAM['model']['decoder'].append(
            {'cell': 'ConvCell', 'input_size': embedding_size,
             'output_size': decoder_hidden_size * ((2 * scale_factor) ** depth),
             'kernel_size': 1, 'stride': 1, 'padding': 0})
        for i in range(depth):
            config.PARAM['model']['decoder'].append({'cell': 'ShuffleCell', 'mode': 'up', 'scale_factor': scale_factor})
            config.PARAM['model']['decoder'].append(
                {'cell': 'ResConvCell', 'input_size': decoder_hidden_size * ((2 * scale_factor) ** (depth - i - 1)),
                 'output_size': decoder_hidden_size * ((2 * scale_factor) ** (depth - i - 1)),
                 'kernel_size': 3, 'stride': 1, 'padding': 1})
        config.PARAM['model']['decoder'].append(
            {'cell': 'ConvCell', 'input_size': decoder_hidden_size, 'output_size': channel_size, 'kernel_size': 1,
             'stride': 1, 'padding': 0, 'activation': 'none'})
        config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    else:
        config.PARAM['model']['decoder'] = []
        config.PARAM['model']['decoder'].append(
            {'cell': 'ConvCell', 'input_size': embedding_size,
             'output_size': decoder_hidden_size * ((2 * scale_factor) ** depth),
             'kernel_size': 1, 'stride': 1, 'padding': 0})
        config.PARAM['model']['decoder'].append(
            {'cell': 'ShuffleCell', 'mode': 'up', 'scale_factor': scale_factor ** depth})
        config.PARAM['model']['decoder'].append(
            {'cell': 'ConvCell', 'input_size': decoder_hidden_size, 'output_size': channel_size, 'kernel_size': 5,
             'stride': 1, 'padding': 2, 'activation': 'none'})
        if split_mode_model == 1:
            config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
        else:
            config.PARAM['model']['decoder'] = [tuple(config.PARAM['model']['decoder']) for _ in range(split_encoder)]
    model = Model()
    return model