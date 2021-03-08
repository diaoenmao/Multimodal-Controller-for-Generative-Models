import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from modules import Wrapper, MultimodalController
from .utils import init_param, make_SpectralNormalization


class GenResBlock(nn.Module):
    def __init__(self, input_size, output_size, num_mode, controller_rate, stride):
        super().__init__()
        self.mc_1 = MultimodalController(input_size, num_mode, controller_rate)
        self.mc_2 = MultimodalController(output_size, num_mode, controller_rate)
        self.conv = nn.Sequential(
            Wrapper(nn.BatchNorm2d(input_size)),
            Wrapper(nn.ReLU()),
            Wrapper(nn.Upsample(scale_factor=stride, mode='nearest')),
            self.mc_1,
            Wrapper(nn.Conv2d(input_size, output_size, 3, 1, 1)),
            Wrapper(nn.BatchNorm2d(output_size)),
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
            self.shortcut = nn.Sequential()

    def forward(self, input):
        shortcut = self.shortcut(input)
        x = self.conv(input)
        x[0] = x[0] + shortcut[0]
        output = x
        return output


class Generator(nn.Module):
    def __init__(self, data_shape, latent_size, hidden_size, num_mode, controller_rate):
        super().__init__()
        self.latent_size = latent_size
        self.linear = Wrapper(nn.Linear(latent_size, hidden_size[0] * 4 * 4))
        blocks = []
        for i in range(len(hidden_size) - 1):
            blocks.append(GenResBlock(hidden_size[i], hidden_size[i + 1], num_mode, controller_rate, stride=2))
        blocks.extend([
            Wrapper(nn.BatchNorm2d(hidden_size[-1])),
            Wrapper(nn.ReLU()),
            MultimodalController(hidden_size[-1], num_mode, controller_rate),
            Wrapper(nn.Conv2d(hidden_size[-1], data_shape[0], 3, 1, 1)),
            Wrapper(nn.Tanh())
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input, indicator):
        x = input
        x = self.linear((x, indicator))
        x[0] = x[0].view(x[0].size(0), -1, 4, 4)
        generated = self.blocks(x)[0]
        return generated


class FirstDisResBlock(nn.Module):
    def __init__(self, input_size, output_size, num_mode, controller_rate):
        super().__init__()
        self.mc_1 = MultimodalController(output_size, num_mode, controller_rate)
        self.conv = nn.Sequential(
            Wrapper(nn.Conv2d(input_size, output_size, 3, 1, 1)),
            Wrapper(nn.ReLU()),
            self.mc_1,
            Wrapper(nn.Conv2d(output_size, output_size, 3, 1, 1)),
            Wrapper(nn.AvgPool2d(2)),
        )
        self.shortcut = nn.Sequential(
            Wrapper(nn.Conv2d(input_size, output_size, 1, 1, 0)),
            Wrapper(nn.AvgPool2d(2)),
        )

    def forward(self, input):
        shortcut = self.shortcut(input)
        x = self.conv(input)
        x[0] = x[0] + shortcut[0]
        output = x
        return output


class DisResBlock(nn.Module):
    def __init__(self, input_size, output_size, num_mode, controller_rate, stride):
        super().__init__()
        self.mc_1 = MultimodalController(input_size, num_mode, controller_rate)
        self.mc_2 = MultimodalController(output_size, num_mode, controller_rate)
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
                self.shortcut = nn.Sequential()

    def forward(self, input):
        shortcut = self.shortcut(input)
        x = self.conv(input)
        x[0] = x[0] + shortcut[0]
        output = x
        return output


class GlobalSumPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        x = input.sum(dim=[-2, -1]).view(input.size(0), -1)
        return x


class Discriminator(nn.Module):
    def __init__(self, data_shape, hidden_size, num_mode, controller_rate):
        super().__init__()
        self.data_shape = data_shape
        blocks = [FirstDisResBlock(data_shape[0], hidden_size[0], num_mode, controller_rate)]
        if cfg['data_name'] in ['CIFAR10', 'CIFAR100']:
            for i in range(len(hidden_size) - 3):
                blocks.append(DisResBlock(hidden_size[i], hidden_size[i + 1], num_mode, controller_rate, stride=2))
            blocks.extend([
                DisResBlock(hidden_size[-3], hidden_size[-2], num_mode, controller_rate, stride=1),
                DisResBlock(hidden_size[-2], hidden_size[-1], num_mode, controller_rate, stride=1),
                Wrapper(nn.ReLU()),
                MultimodalController(hidden_size[-1], num_mode, controller_rate),
                Wrapper(GlobalSumPooling()),
                Wrapper(nn.Linear(hidden_size[-1], 1))
            ])
        else:
            for i in range(len(hidden_size) - 2):
                blocks.append(DisResBlock(hidden_size[i], hidden_size[i + 1], num_mode, controller_rate, stride=2))
            blocks.extend([
                DisResBlock(hidden_size[-2], hidden_size[-1], num_mode, controller_rate, stride=1),
                Wrapper(nn.ReLU()),
                MultimodalController(hidden_size[-1], num_mode, controller_rate),
                Wrapper(GlobalSumPooling()),
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
        self.latent_size = latent_size
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
        x = torch.randn(input['img'].size(0), self.latent_size, device=cfg['device'])
        x = self.generate(input['label'], x)
        x = self.discriminate(x, input['label'])
        return x


def mcgan():
    data_shape = cfg['data_shape']
    latent_size = cfg['gan']['latent_size']
    generator_hidden_size = cfg['gan']['generator_hidden_size']
    discriminator_hidden_size = cfg['gan']['discriminator_hidden_size']
    num_mode = cfg['classes_size']
    controller_rate = cfg['controller_rate']
    model = MCGAN(data_shape, latent_size, generator_hidden_size, discriminator_hidden_size, num_mode, controller_rate)
    model.apply(init_param)
    return model
