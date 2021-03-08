import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param, make_SpectralNormalization


class GenResBlock(nn.Module):
    def __init__(self, input_size, output_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.ReLU(),
            nn.Upsample(scale_factor=stride, mode='nearest'),
            nn.Conv2d(input_size, output_size, 3, 1, 1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
            nn.Conv2d(output_size, output_size, 3, 1, 1),
        )
        if stride > 1:
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='nearest'),
                nn.Conv2d(input_size, output_size, 1, 1, 0)
            )
        elif input_size != output_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_size, output_size, 1, 1, 0)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, input):
        shortcut = self.shortcut(input)
        x = self.conv(input)
        output = x + shortcut
        return output


class Generator(nn.Module):
    def __init__(self, data_shape, latent_size, hidden_size, num_mode, embedding_size):
        super().__init__()
        self.latent_size = latent_size
        self.embedding = nn.Linear(num_mode, embedding_size, bias=False)
        self.linear = nn.Linear(latent_size + embedding_size, hidden_size[0] * 4 * 4)
        blocks = []
        for i in range(len(hidden_size) - 1):
            blocks.append(GenResBlock(hidden_size[i], hidden_size[i + 1], 2))
        blocks.extend([
            nn.BatchNorm2d(hidden_size[-1]),
            nn.ReLU(),
            nn.Conv2d(hidden_size[-1], data_shape[0], 3, 1, 1),
            nn.Tanh()
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input, indicator):
        x = input
        embedding = self.embedding(indicator)
        x = torch.cat((x, embedding), dim=1)
        x = self.linear(x)
        x = x.view(x.size(0), -1, 4, 4)
        generated = self.blocks(x)
        return generated


class DisResBlock(nn.Module):
    def __init__(self, input_size, output_size, stride):
        super().__init__()
        if stride > 1:
            self.conv = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(input_size, output_size, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(output_size, output_size, 3, 1, 1),
                nn.AvgPool2d(2, stride=stride, padding=0),
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_size, output_size, 1, 1, 0),
                nn.AvgPool2d(2, stride=stride, padding=0),
            )
        else:
            self.conv = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(input_size, output_size, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(output_size, output_size, 3, 1, 1),
            )
            if input_size != output_size:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(input_size, output_size, 1, 1, 0)
                )
            else:
                self.shortcut = nn.Sequential()

    def forward(self, input):
        shortcut = self.shortcut(input)
        x = self.conv(input)
        output = x + shortcut
        return output


class FirstDisResBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_size, output_size, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(output_size, output_size, 3, 1, 1),
            nn.AvgPool2d(2),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_size, output_size, 1, 1, 0),
            nn.AvgPool2d(2),
        )

    def forward(self, input):
        x = input
        shortcut = self.shortcut(x)
        x = self.conv(x)
        x = x + shortcut
        output = x
        return output


class GlobalSumPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        x = input.sum(dim=[-2, -1]).view(input.size(0), -1)
        return x


class Discriminator(nn.Module):
    def __init__(self, data_shape, hidden_size, num_mode, embedding_size):
        super().__init__()
        self.data_shape = data_shape
        self.embedding = nn.Linear(num_mode, embedding_size, bias=False)
        blocks = [FirstDisResBlock(data_shape[0] + embedding_size, hidden_size[0])]
        if cfg['data_name'] in ['CIFAR10']:
            for i in range(len(hidden_size) - 3):
                blocks.append(DisResBlock(hidden_size[i], hidden_size[i + 1], stride=2))
            blocks.extend([
                DisResBlock(hidden_size[-3], hidden_size[-2], stride=1),
                DisResBlock(hidden_size[-2], hidden_size[-1], stride=1),
                nn.ReLU(),
                GlobalSumPooling(),
                nn.Linear(hidden_size[-1], 1)
            ])
        else:
            for i in range(len(hidden_size) - 2):
                blocks.append(DisResBlock(hidden_size[i], hidden_size[i + 1], stride=2))
            blocks.extend([
                DisResBlock(hidden_size[-2], hidden_size[-1], stride=1),
                nn.ReLU(),
                GlobalSumPooling(),
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
        self.latent_size = latent_size
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
        x = torch.randn(input['img'].size(0), self.latent_size, device=cfg['device'])
        x = self.generate(input['label'], x)
        x = self.discriminate(x, input['label'])
        return x


def cgan():
    data_shape = cfg['data_shape']
    latent_size = cfg['gan']['latent_size']
    generator_hidden_size = cfg['gan']['generator_hidden_size']
    discriminator_hidden_size = cfg['gan']['discriminator_hidden_size']
    num_mode = cfg['classes_size']
    embedding_size = cfg['gan']['embedding_size']
    model = CGAN(data_shape, latent_size, generator_hidden_size, discriminator_hidden_size, num_mode, embedding_size)
    model.apply(init_param)
    return model