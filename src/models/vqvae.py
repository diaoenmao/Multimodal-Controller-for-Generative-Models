import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import make_model


class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        self.model = make_model(cfg['model'])

    def encode(self, input):
        x = input['img']
        x = self.model['encoder']((x,))[0]
        _, _, code = self.model['quantizer'](x)
        return code

    def decode(self, code):
        x = self.model['quantizer'].embedding_code(code).permute(0, 3, 1, 2).contiguous()
        generated = self.model['decoder']((x,))[0]
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        x = self.model['encoder']((x,))[0]
        x, vq_loss, output['idx'] = self.model['quantizer'](x)
        decoded = self.model['decoder']((x,))[0]
        output['img'] = decoded
        output['loss'] = F.mse_loss(decoded, input['img']) + cfg['vq_commit'] * vq_loss
        return output


def vqvae():
    normalization = 'bn'
    activation = 'relu'
    img_shape = cfg['img_shape']
    hidden_size = cfg['hidden_size']
    quantizer_embedding_size = cfg['quantizer_embedding_size']
    num_embedding = cfg['num_embedding']
    cfg['model'] = {}
    # Encoder
    input_size = img_shape[0]
    output_size = hidden_size
    cfg['model']['encoder'] = []
    cfg['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size
    output_size = hidden_size
    cfg['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size
    output_size = hidden_size
    for i in range(2):
        cfg['model']['encoder'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
             'normalization': normalization, 'activation': activation})
    input_size = hidden_size
    output_size = quantizer_embedding_size
    cfg['model']['encoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': 'none',
         'activation': 'none'})
    cfg['model']['encoder'] = tuple(cfg['model']['encoder'])
    # Quantizer
    cfg['model']['quantizer'] = {
        'cell': 'VectorQuantization', 'embedding_dim': quantizer_embedding_size, 'num_embedding': num_embedding}
    # Decoder
    cfg['model']['decoder'] = []
    input_size = quantizer_embedding_size
    output_size = hidden_size
    cfg['model']['decoder'].append(
        {'cell': 'Conv2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size
    output_size = hidden_size
    for i in range(2):
        cfg['model']['decoder'].append(
            {'cell': 'ResConv2dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
             'normalization': normalization, 'activation': activation})
    input_size = hidden_size
    output_size = hidden_size
    cfg['model']['decoder'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = hidden_size
    output_size = img_shape[0]
    cfg['model']['decoder'].append(
        {'cell': 'ConvTranspose2dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': 'none',
         'activation': 'tanh'})
    cfg['model']['decoder'] = tuple(cfg['model']['decoder'])
    model = VQVAE()
    return model