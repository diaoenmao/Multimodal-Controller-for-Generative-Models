import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from modules import Wrapper, MultimodalController
from .utils import init_param, make_SpectralNormalization

Normalization = nn.BatchNorm2d
Activation = nn.ReLU


class GenResBlock(nn.Module):
    def __init__(self, input_size, output_size, num_mode, controller_rate, mode='pass'):
        super().__init__()
        self.mode = mode
        self.conv = nn.Sequential(
            Wrapper(nn.Conv2d(input_size, output_size, 3, 1, 1)),
            Wrapper(Normalization(output_size)),
            Wrapper(Activation(inplace=True)),
            MultimodalController(output_size, num_mode, controller_rate),
            Wrapper(nn.Conv2d(output_size, output_size, 3, 1, 1)),
            Wrapper(Normalization(output_size)),
            MultimodalController(output_size, num_mode, controller_rate),
        )
        if input_size != output_size:
            self.shortcut = Wrapper(nn.Conv2d(input_size, output_size, 1, 1, 0))
        else:
            self.shortcut = Wrapper(nn.Identity())
        self.activation = Wrapper(Activation(inplace=True))

    def forward(self, input):
        x = (F.interpolate(input[0], scale_factor=2, mode='bilinear',
                           align_corners=False), *input[1:]) if self.mode == 'up' else input
        shortcut = self.shortcut(x)
        x = self.conv(x)
        x[0] = x[0] + shortcut[0]
        x = self.activation(x)
        x[0] = F.avg_pool2d(x[0], 2) if self.mode == 'down' else x[0]
        output = x
        return output


class Generator(nn.Module):
    def __init__(self, data_shape, latent_size, hidden_size, num_mode, controller_rate):
        super().__init__()
        self.latent_size = latent_size
        blocks = [Wrapper(nn.ConvTranspose2d(latent_size, hidden_size[0], 4, 1, 0)),
                  Wrapper(Normalization(hidden_size[0])),
                  Wrapper(Activation(inplace=True))]
        for i in range(len(hidden_size) - 1):
            blocks.append(GenResBlock(hidden_size[i], hidden_size[i + 1], num_mode, controller_rate, mode='up'))
        blocks.extend([
            Wrapper(nn.Conv2d(hidden_size[-1], data_shape[0], 3, 1, 1)),
            Wrapper(nn.Tanh())
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input, indicator):
        x = input
        x = x.view([*x.size(), 1, 1])
        generated = self.blocks((x, indicator))[0]
        return generated


class DisResBlock(nn.Module):
    def __init__(self, input_size, output_size, num_mode, controller_rate, mode='pass'):
        super().__init__()
        self.mode = mode
        self.conv = nn.Sequential(
            Wrapper(nn.Conv2d(input_size, output_size, 3, 1, 1)),
            Wrapper(Activation(inplace=True)),
            MultimodalController(output_size, num_mode, controller_rate),
            Wrapper(nn.Conv2d(output_size, output_size, 3, 1, 1)),
            MultimodalController(output_size, num_mode, controller_rate),
        )
        if input_size != output_size:
            self.shortcut = Wrapper(nn.Conv2d(input_size, output_size, 1, 1, 0))
        else:
            self.shortcut = Wrapper(nn.Identity())
        self.activation = Wrapper(Activation(inplace=True))

    def forward(self, input):
        x = (F.interpolate(input[0], scale_factor=2, mode='bilinear',
                           align_corners=False), *input[1:]) if self.mode == 'up' else input
        shortcut = self.shortcut(x)
        x = self.conv(x)
        x[0] = x[0] + shortcut[0]
        x = self.activation(x)
        x[0] = F.avg_pool2d(x[0], 2) if self.mode == 'down' else x[0]
        output = x
        return output


class Discriminator(nn.Module):
    def __init__(self, data_shape, hidden_size, num_mode, controller_rate):
        super().__init__()
        self.data_shape = data_shape
        blocks = [DisResBlock(data_shape[0], hidden_size[0], num_mode, controller_rate, mode='down')]
        for i in range(len(hidden_size) - 2):
            blocks.append(DisResBlock(hidden_size[i], hidden_size[i + 1], num_mode, controller_rate, mode='down'))
        blocks.extend([
            DisResBlock(hidden_size[-2], hidden_size[-1], num_mode, controller_rate, mode='pass'),
            Wrapper(nn.AdaptiveAvgPool2d(1)),
            Wrapper(nn.Flatten()),
            Wrapper(nn.Linear(hidden_size[-1], 1))
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input, indicator):
        x = input
        x = self.blocks((x, indicator))[0]
        return x


class MCGAN(nn.Module):
    def __init__(self, data_shape, latent_size, generator_hidden_size, discriminator_hidden_size, num_mode,
                 controller_rate):
        super().__init__()
        self.generator = Generator(data_shape, latent_size, generator_hidden_size, num_mode, controller_rate)
        self.discriminator = Discriminator(data_shape, discriminator_hidden_size, num_mode, controller_rate)
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


def mcgan():
    data_shape = cfg['data_shape']
    latent_size = cfg['latent_size']
    generator_hidden_size = cfg['generator_hidden_size']
    discriminator_hidden_size = cfg['discriminator_hidden_size']
    num_mode = cfg['classes_size']
    controller_rate = cfg['controller_rate']
    model = MCGAN(data_shape, latent_size, generator_hidden_size, discriminator_hidden_size, num_mode, controller_rate)
    model.apply(init_param)
    return model