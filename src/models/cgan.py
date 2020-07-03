import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param, make_SpectralNormalization

Normalization = nn.BatchNorm2d
Activation = nn.ReLU


class GenResBlock(nn.Module):
    def __init__(self, input_size, output_size, mode='pass'):
        super().__init__()
        self.mode = mode
        self.conv = nn.Sequential(
            nn.Conv2d(input_size, output_size, 3, 1, 1),
            Normalization(output_size),
            Activation(inplace=True),
            nn.Conv2d(output_size, output_size, 3, 1, 1),
            Normalization(output_size),
        )
        if input_size != output_size:
            self.shortcut = nn.Conv2d(input_size, output_size, 1, 1, 0)
        else:
            self.shortcut = nn.Identity()
        self.activation = Activation(inplace=True)

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False) if self.mode == 'up' else input
        shortcut = self.shortcut(x)
        x = self.conv(x)
        x = self.activation(x + shortcut)
        output = F.avg_pool2d(x, 2) if self.mode == 'down' else x
        return output


class Generator(nn.Module):
    def __init__(self, data_shape, latent_size, hidden_size, num_mode, embedding_size):
        super().__init__()
        self.latent_size = latent_size
        self.embedding = nn.Linear(num_mode, embedding_size, bias=False)
        blocks = [nn.ConvTranspose2d(latent_size + embedding_size, hidden_size[0], 4, 1, 0),
                  Normalization(hidden_size[0]),
                  Activation(inplace=True)]
        for i in range(len(hidden_size) - 1):
            blocks.append(GenResBlock(hidden_size[i], hidden_size[i + 1], mode='up'))
        blocks.extend([
            nn.Conv2d(hidden_size[-1], data_shape[0], 3, 1, 1),
            nn.Tanh()
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input, indicator):
        x = input
        embedding = self.embedding(indicator)
        embedding = embedding.view([*embedding.size(), 1, 1])
        x = x.view([*x.size(), 1, 1])
        x = torch.cat((x, embedding), dim=1)
        x = self.blocks(x)
        return x


class DisResBlock(nn.Module):
    def __init__(self, input_size, output_size, mode='pass'):
        super().__init__()
        self.mode = mode
        self.conv = nn.Sequential(
            nn.Conv2d(input_size, output_size, 3, 1, 1),
            Activation(inplace=True),
            nn.Conv2d(output_size, output_size, 3, 1, 1),
        )
        if input_size != output_size:
            self.shortcut = nn.Conv2d(input_size, output_size, 1, 1, 0)
        else:
            self.shortcut = nn.Identity()
        self.activation = Activation(inplace=True)

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False) if self.mode == 'up' else input
        shortcut = self.shortcut(x)
        x = self.conv(x)
        x = self.activation(x + shortcut)
        output = F.avg_pool2d(x, 2) if self.mode == 'down' else x
        return output


class Discriminator(nn.Module):
    def __init__(self, data_shape, hidden_size, num_mode, embedding_size):
        super().__init__()
        self.data_shape = data_shape
        self.embedding = nn.Linear(num_mode, embedding_size, bias=False)
        blocks = [DisResBlock(data_shape[0] + embedding_size, hidden_size[0], mode='down')]
        for i in range(len(hidden_size) - 2):
            blocks.append(DisResBlock(hidden_size[i], hidden_size[i + 1], mode='down'))
        blocks.extend([
            DisResBlock(hidden_size[-2], hidden_size[-1], mode='pass'),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_size[-1], 1)
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input, indicator):
        x = input
        embedding = self.embedding(indicator)
        embedding = embedding.view([*embedding.size(), 1, 1]).expand([*embedding.size(), *x.size()[2:]])
        x = torch.cat((x, embedding), dim=1)
        x = self.blocks(x)
        return x


class CGAN(nn.Module):
    def __init__(self, data_shape, latent_size, generator_hidden_size, discriminator_hidden_size, num_mode,
                 embedding_size):
        super().__init__()
        self.generator = Generator(data_shape, latent_size, generator_hidden_size, num_mode, embedding_size)
        self.discriminator = Discriminator(data_shape, discriminator_hidden_size, num_mode, embedding_size)
        self.discriminator.apply(make_SpectralNormalization)

    def generate(self, C, x=None):
        if x is None:
            x = torch.randn([C.size(0), self.latent_size], device=cfg['device'])
        indicator = F.one_hot(C, cfg['classes_size']).float()
        generated = self.generator(x, indicator)
        return generated

    def discriminate(self, x, C):
        indicator = F.one_hot(C, cfg['classes_size']).float()
        discriminated = self.discriminator(x, indicator)
        return discriminated

    def forward(self, input):
        x = torch.randn(input['img'].size(0), cfg['latent_size'], device=cfg['device'])
        x = self.generate(input['label'], x)
        x = self.discriminate(x, input['label'])
        return x


def cgan():
    data_shape = cfg['data_shape']
    latent_size = cfg['latent_size']
    generator_hidden_size = cfg['generator_hidden_size']
    discriminator_hidden_size = cfg['discriminator_hidden_size']
    num_mode = cfg['classes_size']
    embedding_size = cfg['embedding_size']
    model = CGAN(data_shape, latent_size, generator_hidden_size, discriminator_hidden_size, num_mode, embedding_size)
    model.apply(init_param)
    return model