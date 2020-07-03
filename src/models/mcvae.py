import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from modules import Wrapper, MultimodalController
from .utils import init_param

Normalization = nn.BatchNorm2d
Activation = nn.ReLU


def loss(input, output):
    CE = F.binary_cross_entropy(output['img'], input['img'], reduction='sum')
    KLD = -0.5 * torch.sum(1 + output['logvar'] - output['mu'].pow(2) - output['logvar'].exp())
    return CE + KLD


class ResBlock(nn.Module):
    def __init__(self, hidden_size, num_mode, controller_rate):
        super().__init__()
        self.conv = nn.Sequential(
            Wrapper(nn.Conv2d(hidden_size, hidden_size, 3, 1, 1)),
            Wrapper(Normalization(hidden_size)),
            Wrapper(Activation(inplace=True)),
            MultimodalController(hidden_size, num_mode, controller_rate),
            Wrapper(nn.Conv2d(hidden_size, hidden_size, 3, 1, 1)),
            Wrapper(Normalization(hidden_size)),
            MultimodalController(hidden_size, num_mode, controller_rate),
        )
        self.activation = Wrapper(Activation(inplace=True))

    def forward(self, input):
        x = self.conv(input)
        x[0] = x[0] + input[0]
        output = self.activation(x)
        return output


class Encoder(nn.Module):
    def __init__(self, data_shape, hidden_size, latent_size, num_res_block, num_mode, controller_rate):
        super().__init__()
        blocks = [
            Wrapper(nn.Conv2d(data_shape[0], hidden_size[0], 4, 2, 1)),
            Wrapper(Normalization(hidden_size[0])),
            Wrapper(Activation(inplace=True)),
            MultimodalController(hidden_size[0], num_mode, controller_rate),
            Wrapper(nn.Conv2d(hidden_size[0], hidden_size[1], 4, 2, 1)),
            Wrapper(Normalization(hidden_size[1])),
            Wrapper(Activation(inplace=True)),
            MultimodalController(hidden_size[1], num_mode, controller_rate),
            Wrapper(nn.Conv2d(hidden_size[1], hidden_size[2], 4, 2, 1)),
            Wrapper(Normalization(hidden_size[2])),
            Wrapper(Activation(inplace=True)),
            MultimodalController(hidden_size[2], num_mode, controller_rate),
        ]
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size[2], num_mode, controller_rate))
        self.blocks = nn.Sequential(*blocks)
        self.encoded_shape = (hidden_size[2], data_shape[1] // (2 ** len(hidden_size)),
                              data_shape[2] // (2 ** len(hidden_size)))
        self.mu = nn.Linear(np.prod(self.encoded_shape).item(), latent_size)
        self.logvar = nn.Linear(np.prod(self.encoded_shape).item(), latent_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input):
        x = self.blocks(input)[0]
        x = x.view(x.size(0), -1)
        mu, logvar = self.mu(x), self.logvar(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, data_shape, hidden_size, latent_size, num_res_block, num_mode, controller_rate):
        super().__init__()
        self.encoded_shape = (hidden_size[2], data_shape[1] // (2 ** len(hidden_size)),
                              data_shape[2] // (2 ** len(hidden_size)))
        self.linear = nn.Sequential(
            MultimodalController(latent_size, num_mode, controller_rate),
            Wrapper(nn.Linear(latent_size, np.prod(self.encoded_shape).item())),
            Wrapper(nn.BatchNorm1d(np.prod(self.encoded_shape).item())),
            Wrapper(Activation(inplace=True)),
        )
        blocks = [MultimodalController(hidden_size[2], num_mode, controller_rate)]
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size[2], num_mode, controller_rate))
        blocks.extend([
            Wrapper(nn.ConvTranspose2d(hidden_size[2], hidden_size[1], 4, 2, 1)),
            Wrapper(Normalization(hidden_size[1])),
            Wrapper(Activation(inplace=True)),
            MultimodalController(hidden_size[1], num_mode, controller_rate),
            Wrapper(nn.ConvTranspose2d(hidden_size[1], hidden_size[0], 4, 2, 1)),
            Wrapper(Normalization(hidden_size[0])),
            Wrapper(Activation(inplace=True)),
            MultimodalController(hidden_size[0], num_mode, controller_rate),
            Wrapper(nn.ConvTranspose2d(hidden_size[0], data_shape[0], 4, 2, 1)),
            Wrapper(nn.Sigmoid())
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        x = self.linear(input)
        x[0] = x[0].view(x[0].size(0), *self.encoded_shape)
        x = self.blocks(x)[0]
        return x


class MCVAE(nn.Module):
    def __init__(self, data_shape=(3, 32, 32), hidden_size=(64, 128, 256), latent_size=128, num_res_block=2,
                 num_mode=None, controller_rate=0.5):
        super().__init__()
        self.data_shape = data_shape
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_res_block = num_res_block
        self.num_mode = num_mode
        self.controller_rate = controller_rate
        self.encoder = Encoder(data_shape, hidden_size, latent_size, num_res_block, num_mode, controller_rate)
        self.decoder = Decoder(data_shape, hidden_size, latent_size, num_res_block, num_mode, controller_rate)

    def encode(self, input):
        z, mu, logvar = self.encoder(input)
        return z, mu, logvar

    def decode(self, input):
        decoded = self.decoder(input)
        return decoded

    def generate(self, C, z=None):
        if z is None:
            z = torch.randn([C.size(0), self.latent_size], device=cfg['device'])
        indicator = F.one_hot(C, cfg['classes_size']).float()
        x = self.decode([z, indicator])
        generated = x * 2 - 1
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        input['img'] = (input['img'] + 1) / 2
        x = input['img']
        indicator = F.one_hot(input['label'], cfg['classes_size']).float()
        z, output['mu'], output['logvar'] = self.encode([x, indicator])
        decoded = self.decode([z, indicator])
        output['img'] = decoded
        output['loss'] = loss(input, output)
        input['img'] = input['img'] * 2 - 1
        output['img'] = output['img'] * 2 - 1
        return output


def mcvae():
    data_shape = cfg['data_shape']
    hidden_size = cfg['hidden_size']
    latent_size = cfg['latent_size']
    num_res_block = cfg['num_res_block']
    num_mode = cfg['classes_size']
    controller_rate = cfg['controller_rate']
    model = MCVAE(data_shape=data_shape, hidden_size=hidden_size, latent_size=latent_size, num_res_block=num_res_block,
                  num_mode=num_mode, controller_rate=controller_rate)
    model.apply(init_param)
    return model