import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from modules import Wrapper, MultimodalController
from .utils import init_param, make_SpectralNormalization


class GenResBlock(nn.Module):
    def __init__(self, input_size, output_size, num_mode, stride):
        super().__init__()
        self.mc_1 = MultimodalController(input_size, num_mode)
        self.mc_2 = MultimodalController(output_size, num_mode)
        self.conv = nn.Sequential(
            # Wrapper(nn.BatchNorm2d(input_size)),
            Wrapper(nn.ReLU()),
            Wrapper(nn.Upsample(scale_factor=stride, mode='nearest')),
            self.mc_1,
            Wrapper(nn.Conv2d(input_size, output_size, 3, 1, 1)),
            # Wrapper(nn.BatchNorm2d(output_size)),
            Wrapper(nn.ReLU()),
            self.mc_2,
            Wrapper(nn.Conv2d(output_size, output_size, 3, 1, 1)),
        )
        if stride > 1:
            self.shortcut = nn.Sequential(
                Wrapper(nn.Upsample(scale_factor=stride, mode='nearest')),
                self.mc_1,
                Wrapper(nn.Conv2d(input_size, output_size, 1, 1, 0))
            )
        elif input_size != output_size:
            self.shortcut = nn.Sequential(
                self.mc_1,
                nn.Conv2d(input_size, output_size, 1, 1, 0)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, input):
        shortcut = self.shortcut(input)
        x = self.conv(input)
        x[0] = x[0] + shortcut[0]
        return x


class Generator(nn.Module):
    def __init__(self, data_shape, latent_size, hidden_size, num_mode):
        super().__init__()
        self.latent_size = latent_size
        self.linear = Wrapper(nn.Linear(latent_size, hidden_size[0] * 4 * 4))
        blocks = []
        for i in range(len(hidden_size) - 1):
            blocks.append(GenResBlock(hidden_size[i], hidden_size[i + 1], num_mode, stride=2))
        blocks.extend([
            # Wrapper(nn.BatchNorm2d(hidden_size[-1])),
            Wrapper(nn.ReLU()),
            MultimodalController(hidden_size[-1], num_mode),
            Wrapper(nn.Conv2d(hidden_size[-1], data_shape[0], 3, 1, 1)),
            Wrapper(nn.Tanh())
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input, label):
        x = input
        x = self.linear((x, label))
        x[0] = x[0].view(x[0].size(0), -1, 4, 4)
        x = self.blocks(x)[0]
        return x


class FirstDisResBlock(nn.Module):
    def __init__(self, input_size, output_size, num_mode):
        super().__init__()
        self.mc_1 = MultimodalController(output_size, num_mode)
        self.conv = nn.Sequential(
            Wrapper(nn.Conv2d(input_size, output_size, 3, 1, 1)),
            Wrapper(nn.ReLU()),
            self.mc_1,
            Wrapper(nn.Conv2d(output_size, output_size, 3, 1, 1)),
            Wrapper(nn.AvgPool2d(2)),
        )
        self.shortcut = nn.Sequential(
            Wrapper(nn.AvgPool2d(2)),
            Wrapper(nn.Conv2d(input_size, output_size, 1, 1, 0)),
        )

    def forward(self, input):
        x = input
        shortcut = self.shortcut(x)
        x = self.conv(input)
        x[0] = x[0] + shortcut[0]
        return x


class DisResBlock(nn.Module):
    def __init__(self, input_size, output_size, num_mode, stride):
        super().__init__()
        self.mc_1 = MultimodalController(input_size, num_mode)
        self.mc_2 = MultimodalController(output_size, num_mode)
        if stride > 1:
            self.conv = nn.Sequential(
                Wrapper(nn.ReLU()),
                self.mc_1,
                Wrapper(nn.Conv2d(input_size, output_size, 3, 1, 1)),
                Wrapper(nn.ReLU()),
                self.mc_2,
                Wrapper(nn.Conv2d(output_size, output_size, 3, 1, 1)),
                Wrapper(nn.AvgPool2d(2)),
            )
            self.shortcut = nn.Sequential(
                self.mc_1,
                Wrapper(nn.Conv2d(input_size, output_size, 1, 1, 0)),
                Wrapper(nn.AvgPool2d(2)),
            )
        else:
            self.conv = nn.Sequential(
                Wrapper(nn.ReLU()),
                self.mc_1,
                Wrapper(nn.Conv2d(input_size, output_size, 3, 1, 1)),
                Wrapper(nn.ReLU()),
                self.mc_2,
                Wrapper(nn.Conv2d(output_size, output_size, 3, 1, 1)),
            )
            if input_size != output_size:
                self.shortcut = nn.Sequential(
                    self.mc_1,
                    Wrapper(nn.Conv2d(input_size, output_size, 1, 1, 0))
                )
            else:
                self.shortcut = nn.Identity()

    def forward(self, input):
        shortcut = self.shortcut(input)
        x = self.conv(input)
        x[0] = x[0] + shortcut[0]
        return x


class Discriminator(nn.Module):
    def __init__(self, data_shape, hidden_size, num_mode):
        super().__init__()
        self.data_shape = data_shape
        blocks = [FirstDisResBlock(data_shape[0], hidden_size[0], num_mode)]
        for i in range(len(hidden_size) - 3):
            blocks.append(DisResBlock(hidden_size[i], hidden_size[i + 1], num_mode, stride=2))
        blocks.extend([
            DisResBlock(hidden_size[-3], hidden_size[-2], num_mode, stride=1),
            DisResBlock(hidden_size[-2], hidden_size[-1], num_mode, stride=1),
            Wrapper(nn.ReLU()),
            MultimodalController(hidden_size[-1], num_mode),
            Wrapper(nn.AdaptiveAvgPool2d(1)),
            Wrapper(nn.Flatten()),
            Wrapper(nn.Linear(hidden_size[-1], 1))
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input, label):
        x = input
        x = self.blocks((x, label))[0]
        return x


class MCGAN(nn.Module):
    def __init__(self, data_shape, latent_size, generator_hidden_size, discriminator_hidden_size, num_mode):
        super().__init__()
        self.latent_size = latent_size
        self.generator = Generator(data_shape, latent_size, generator_hidden_size, num_mode)
        self.discriminator = Discriminator(data_shape, discriminator_hidden_size, num_mode)
        self.generator.apply(make_SpectralNormalization)
        self.discriminator.apply(make_SpectralNormalization)

    def forward(self, input):
        x = torch.randn(input['data'].size(0), cfg['mcgan']['latent_size'], device=cfg['device'])
        x = self.generator(x, input['target'])
        x = self.discriminator(x, input['target'])
        return x


def mcgan():
    data_shape = cfg['data_shape']
    latent_size = cfg['mcgan']['latent_size']
    generator_hidden_size = cfg['mcgan']['generator_hidden_size']
    discriminator_hidden_size = cfg['mcgan']['discriminator_hidden_size']
    num_mode = cfg['target_size']
    model = MCGAN(data_shape, latent_size, generator_hidden_size, discriminator_hidden_size, num_mode)
    model.apply(init_param)
    return model
