import config
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import make_model, make_SpectralNormalization


class GResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, activation='relu', upsample=False):
        super(GResBlock, self).__init__()
        self.upsample = upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.conv_1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.activation = make_model({'cell': 'Activation', 'mode': activation})
        self.bn_1 = nn.BatchNorm2d(in_channels)
        self.bn_2 = nn.BatchNorm2d(hidden_channels)
        if in_channels != out_channels or self.upsample:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.conv_sc = nn.Identity()

    def residual(self, x):
        x = self.bn_1(x)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv_1(x)
        x = self.bn_2(x)
        x = self.activation(x)
        x = self.conv_2(x)
        return x

    def shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv_sc(x)
        return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, activation='leakyrelu', downsample=False,
                 init=False):
        super(DResBlock, self).__init__()
        self.downsample = downsample
        self.init = init
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.conv_1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.activation = make_model({'cell': 'Activation', 'mode': activation})
        if in_channels != out_channels or self.downsample:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.conv_sc = nn.Identity()

    def residual(self, x):
        if not self.init:
            x = self.activation(x)
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        if self.init:
            if self.downsample:
                x = F.avg_pool2d(x, 2)
            x = self.conv_sc(x)
        else:
            x = self.conv_sc(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class MCGResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, activation='relu', upsample=False,
                 num_mode=10, controller_rate=0.5):
        super(MCGResBlock, self).__init__()
        self.upsample = upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.conv_1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.activation = make_model({'cell': 'Activation', 'mode': activation})
        self.bn_1 = nn.BatchNorm2d(in_channels)
        self.bn_2 = nn.BatchNorm2d(hidden_channels)
        self.mc_1 = make_model({'cell': 'MultimodalController', 'input_size': in_channels, 'num_mode': num_mode,
                                'controller_rate': controller_rate})
        self.mc_2 = make_model({'cell': 'MultimodalController', 'input_size': hidden_channels, 'num_mode': num_mode,
                                'controller_rate': controller_rate})
        if in_channels != out_channels or self.upsample:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.conv_sc = nn.Identity()

    def residual(self, x):
        x = self.bn_1(x)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.mc_1(x)
        x = self.conv_1(x)
        x = self.bn_2(x)
        x = self.activation(x)
        x = self.mc_2(x)
        x = self.conv_2(x)
        return x

    def shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.mc_2(x)
        x = self.conv_sc(x)
        return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class MCDResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, activation='leakyrelu', num_mode=10,
                 controller_rate=0.5, downsample=False, init=False):
        super(MCDResBlock, self).__init__()
        self.downsample = downsample
        self.init = init
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.conv_1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.activation = make_model({'cell': 'Activation', 'mode': activation})
        self.mc_1 = make_model({'cell': 'MultimodalController', 'input_size': in_channels, 'num_mode': num_mode,
                                'controller_rate': controller_rate})
        self.mc_2 = make_model({'cell': 'MultimodalController', 'input_size': hidden_channels, 'num_mode': num_mode,
                                'controller_rate': controller_rate})
        if in_channels != out_channels or self.downsample:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.conv_sc = nn.Identity()

    def residual(self, x):
        if not self.init:
            x = self.activation(x)
            x = self.mc_1(x)
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.mc_2(x)
        x = self.conv_2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        if self.init:
            if self.downsample:
                x = F.avg_pool2d(x, 2)
            x = self.conv_sc(x)
        else:
            x = self.mc_1(x)
            x = self.conv_sc(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class CGenerator(nn.Module):
    def __init__(self, latent_size, hidden_size, conditional_embedding_size, encode_shape, output_size, activation,
                 num_mode):
        super(CGenerator, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.encode_shape = encode_shape
        self.output_size = output_size
        self.e_0 = nn.Linear(num_mode, conditional_embedding_size, bias=False)
        self.l_1 = nn.Linear(self.latent_size + conditional_embedding_size, np.prod(self.encode_shape))
        self.block_2 = GResBlock(self.hidden_size, self.hidden_size, activation=activation, upsample=True)
        self.block_3 = GResBlock(self.hidden_size, self.hidden_size, activation=activation, upsample=True)
        self.block_4 = GResBlock(self.hidden_size, self.hidden_size, activation=activation, upsample=True)
        self.b_5 = nn.BatchNorm2d(self.hidden_size)
        self.activation = make_model({'cell': 'Activation', 'mode': activation})
        self.conv_5 = nn.Conv2d(self.hidden_size, self.output_size, kernel_size=3, padding=1)

    def forward(self, x, C):
        e = self.e_0(C)
        x = torch.cat((x, e), dim=1)
        x = self.l_1(x).view(x.size(0), *self.encode_shape)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.activation(self.b_5(x))
        x = torch.tanh(self.conv_5(x))
        return x


class CDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, conditional_embedding_size, activation, num_mode):
        super(CDiscriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.e_0 = nn.Linear(num_mode, conditional_embedding_size, bias=False)
        self.block_1 = DResBlock(self.input_size + conditional_embedding_size, self.hidden_size, downsample=True,
                                 init=True)
        self.block_2 = DResBlock(self.hidden_size, self.hidden_size, activation=activation, downsample=True)
        self.block_3 = DResBlock(self.hidden_size, self.hidden_size, activation=activation, downsample=False)
        self.block_4 = DResBlock(self.hidden_size, self.hidden_size, activation=activation, downsample=False)
        self.activation = make_model({'cell': 'Activation', 'mode': activation})
        self.l_5 = nn.Linear(self.hidden_size, 1)

    def forward(self, x, C):
        e = self.e_0(C)
        e = e.view([*e.size(), 1, 1]).expand([*e.size(), *x.size()[2:]])
        x = torch.cat((x, e), dim=1)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        x = self.l_5(x)
        return x


class MCGenerator(nn.Module):
    def __init__(self, latent_size, hidden_size, encode_shape, output_size, activation, num_mode, controller_rate):
        super(MCGenerator, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.encode_shape = encode_shape
        self.output_size = output_size
        self.l_1 = nn.Linear(self.latent_size, np.prod(self.encode_shape))
        self.block_2 = MCGResBlock(self.hidden_size, self.hidden_size, activation=activation, upsample=True,
                                   num_mode=num_mode, controller_rate=controller_rate)
        self.block_3 = MCGResBlock(self.hidden_size, self.hidden_size, activation=activation, upsample=True,
                                   num_mode=num_mode, controller_rate=controller_rate)
        self.block_4 = MCGResBlock(self.hidden_size, self.hidden_size, activation=activation, upsample=True,
                                   num_mode=num_mode, controller_rate=controller_rate)
        self.b_5 = nn.BatchNorm2d(self.hidden_size)
        self.activation = make_model({'cell': 'Activation', 'mode': activation})
        self.mc_5 = make_model({'cell': 'MultimodalController', 'input_size': self.hidden_size, 'num_mode': num_mode,
                                'controller_rate': controller_rate})
        self.conv_5 = nn.Conv2d(self.hidden_size, self.output_size, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.l_1(x).view(x.size(0), *self.encode_shape)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.mc_5(self.activation(self.b_5(x)))
        x = torch.tanh(self.conv_5(x))
        return x


class MCDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, activation, num_mode, controller_rate):
        super(MCDiscriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.block_1 = DResBlock(self.input_size, self.hidden_size, downsample=True,
                                 init=True)
        self.block_2 = DResBlock(self.hidden_size, self.hidden_size, activation=activation, downsample=True)
        self.block_3 = DResBlock(self.hidden_size, self.hidden_size, activation=activation, downsample=False)
        self.block_4 = DResBlock(self.hidden_size, self.hidden_size, activation=activation, downsample=False)
        self.activation = make_model({'cell': 'Activation', 'mode': activation})
        self.mc_5 = make_model({'cell': 'MultimodalController', 'input_size': self.hidden_size, 'num_mode': num_mode,
                                'controller_rate': controller_rate})
        self.l_5 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        x = self.mc_5(x)
        x = self.l_5(x)
        return x


class CSNGAN(nn.Module):
    def __init__(self):
        super(CSNGAN, self).__init__()
        self.model = nn.ModuleDict({})
        self.model['generator'] = CGenerator(config.PARAM['latent_size'], config.PARAM['generator_hidden_size'],
                                             config.PARAM['conditional_embedding_size'], config.PARAM['encode_shape'],
                                             config.PARAM['img_shape'][0], config.PARAM['generator_activation'],
                                             config.PARAM['classes_size'])
        self.model['discriminator'] = CDiscriminator(config.PARAM['img_shape'][0],
                                                     config.PARAM['discriminator_hidden_size'],
                                                     config.PARAM['conditional_embedding_size'],
                                                     config.PARAM['discriminator_activation'],
                                                     config.PARAM['classes_size'])
        make_SpectralNormalization(self.model['discriminator'])

    def generate(self, x, C):
        C = F.one_hot(C, config.PARAM['classes_size']).float()
        generated = self.model['generator'](x, C)
        return generated

    def discriminate(self, x, C):
        C = F.one_hot(C, config.PARAM['classes_size']).float()
        discriminated = self.model['discriminator'](x, C)
        return discriminated

    def forward(self, input):
        z = torch.randn(input['img'].size(0), config.PARAM['latent_size'], device=config.PARAM['device'])
        x = self.generate(z, input['label'])
        x = self.discriminate(x, input['label'])
        return x


class MCSNGAN(nn.Module):
    def __init__(self):
        super(MCSNGAN, self).__init__()
        self.model = nn.ModuleDict({})
        self.model['generator'] = MCGenerator(config.PARAM['latent_size'], config.PARAM['generator_hidden_size'],
                                              config.PARAM['encode_shape'], config.PARAM['img_shape'][0],
                                              config.PARAM['generator_activation'], config.PARAM['classes_size'],
                                              config.PARAM['controller_rate'])
        self.model['discriminator'] = MCDiscriminator(config.PARAM['img_shape'][0],
                                                      config.PARAM['discriminator_hidden_size'],
                                                      config.PARAM['discriminator_activation'],
                                                      config.PARAM['classes_size'], config.PARAM['controller_rate'])
        make_SpectralNormalization(self.model['discriminator'])

    def generate(self, x, C):
        config.PARAM['indicator'] = F.one_hot(C, config.PARAM['classes_size']).float()
        generated = self.model['generator'](x)
        return generated

    def discriminate(self, input, C):
        x = input
        config.PARAM['indicator'] = F.one_hot(C, config.PARAM['classes_size']).float()
        discriminated = self.model['discriminator'](x)
        return discriminated

    def forward(self, input):
        z = torch.randn(input['img'].size(0), config.PARAM['latent_size'], device=config.PARAM['device'])
        x = self.generate(z, input['label'])
        x = self.discriminate(x, input['label'])
        return x


def csngan():
    model = CSNGAN()
    return model


def mcsngan():
    model = MCSNGAN()
    return model