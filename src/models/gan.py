import config
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


class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
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