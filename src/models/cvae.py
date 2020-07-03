import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param

Normalization = nn.BatchNorm2d
Activation = nn.ReLU


def loss(input, output):
    CE = F.binary_cross_entropy(output['img'], input['img'], reduction='sum')
    KLD = 0.5 * torch.sum(output['mu'].pow(2) + output['logvar'].exp() - 1 - output['logvar'])
    return CE + KLD


class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.activation = Activation(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            Normalization(hidden_size),
            Activation(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            Normalization(hidden_size),
        )

    def forward(self, input):
        output = self.conv(input)
        output = self.activation(output + input)
        return output


class Encoder(nn.Module):
    def __init__(self, data_shape, hidden_size, latent_size, num_res_block, num_mode, embedding_size):
        super().__init__()
        self.embedding = nn.Linear(num_mode, embedding_size, False)
        blocks = [
            nn.Conv2d(data_shape[0] + embedding_size, hidden_size[0], 4, 2, 1),
            Normalization(hidden_size[0]),
            Activation(inplace=True),
            nn.Conv2d(hidden_size[0], hidden_size[1], 4, 2, 1),
            Normalization(hidden_size[1]),
            Activation(inplace=True),
            nn.Conv2d(hidden_size[1], hidden_size[2], 4, 2, 1),
            Normalization(hidden_size[2]),
            Activation(inplace=True),
        ]
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size[2]))
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
        x, indicator = input
        embedding = self.embedding(indicator)
        embedding = embedding.view([*embedding.size(), 1, 1]).expand([*embedding.size(), *x.size()[2:]])
        x = torch.cat((x, embedding), dim=1)
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        mu, logvar = self.mu(x), self.logvar(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, data_shape, hidden_size, latent_size, num_res_block, num_mode, embedding_size):
        super().__init__()
        self.embedding = nn.Linear(num_mode, embedding_size, False)
        self.encoded_shape = (hidden_size[2], data_shape[1] // (2 ** len(hidden_size)),
                              data_shape[2] // (2 ** len(hidden_size)))
        self.linear = nn.Sequential(
            nn.Linear(latent_size + embedding_size, np.prod(self.encoded_shape).item()),
            nn.BatchNorm1d(np.prod(self.encoded_shape).item()),
            Activation(inplace=True),
        )
        blocks = []
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size[2]))
        blocks.extend([
            nn.ConvTranspose2d(hidden_size[2], hidden_size[1], 4, 2, 1),
            Normalization(hidden_size[1]),
            Activation(inplace=True),
            nn.ConvTranspose2d(hidden_size[1], hidden_size[0], 4, 2, 1),
            Normalization(hidden_size[0]),
            Activation(inplace=True),
            nn.ConvTranspose2d(hidden_size[0], data_shape[0], 4, 2, 1),
            nn.Sigmoid()
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        z, indicator = input
        embedding = self.embedding(indicator)
        x = torch.cat((z, embedding), dim=1)
        x = self.linear(x)
        x = x.view(x.size(0), *self.encoded_shape)
        x = self.blocks(x)
        return x


class CVAE(nn.Module):
    def __init__(self, data_shape=(3, 32, 32), hidden_size=(64, 128, 256), latent_size=128, num_res_block=2,
                 num_mode=None, embedding_size=32):
        super().__init__()
        self.data_shape = data_shape
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_res_block = num_res_block
        self.embedding_size = embedding_size
        self.num_mode = num_mode
        self.encoder = Encoder(data_shape, hidden_size, latent_size, num_res_block, num_mode, embedding_size)
        self.decoder = Decoder(data_shape, hidden_size, latent_size, num_res_block, num_mode, embedding_size)

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
        x = self.decode((z, indicator))
        generated = x * 2 - 1
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        input['img'] = (input['img'] + 1) / 2
        x = input['img']
        indicator = F.one_hot(input['label'], cfg['classes_size']).float()
        z, output['mu'], output['logvar'] = self.encode((x, indicator))
        decoded = self.decode((z, indicator))
        output['img'] = decoded
        output['loss'] = loss(input, output)
        input['img'] = input['img'] * 2 - 1
        output['img'] = output['img'] * 2 - 1
        return output


def cvae():
    data_shape = cfg['data_shape']
    hidden_size = cfg['hidden_size']
    latent_size = cfg['latent_size']
    num_res_block = cfg['num_res_block']
    num_mode = cfg['classes_size']
    embedding_size = cfg['embedding_size']
    model = CVAE(data_shape=data_shape, hidden_size=hidden_size, latent_size=latent_size, num_res_block=num_res_block,
                 num_mode=num_mode, embedding_size=embedding_size)
    model.apply(init_param)
    return model