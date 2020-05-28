import config
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import make_model, make_SpectralNormalization


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1, upsample=False, num_mode=0,
                 do_mc=False):
        super(GenBlock, self).__init__()
        self.activation = nn.ReLU()
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        self.do_mc = do_mc
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.num_mode = num_mode
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        self.b1 = nn.BatchNorm2d(in_channels)
        self.b2 = nn.BatchNorm2d(hidden_channels)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        if do_mc:
            self.mc1 = make_model({'cell': 'MultimodalController', 'input_size': in_channels, 'num_mode': num_mode,
                                   'controller_rate': config.PARAM['controller_rate']})
            self.mc2 = make_model({'cell': 'MultimodalController', 'input_size': hidden_channels, 'num_mode': num_mode,
                                   'controller_rate': config.PARAM['controller_rate']})

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        if self.do_mc:
            h = self.mc1(h)
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        h = self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        if self.do_mc:
            h = self.mc2(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            if self.do_mc:
                x = self.mc1(x)
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Generator(nn.Module):
    def __init__(self, img_shape, latent_size, hidden_size, num_mode=0, do_mc=False):
        super(Generator, self).__init__()
        self.bottom_width = img_shape[1] // 8
        self.activation = nn.ReLU()
        self.num_mode = num_mode
        self.ch = hidden_size
        self.do_mc = do_mc
        self.l1 = nn.Linear(latent_size, (self.bottom_width ** 2) * self.ch)
        self.block2 = GenBlock(self.ch, self.ch, upsample=True, num_mode=num_mode, do_mc=do_mc)
        self.block3 = GenBlock(self.ch, self.ch, upsample=True, num_mode=num_mode, do_mc=do_mc)
        self.block4 = GenBlock(self.ch, self.ch, upsample=True, num_mode=num_mode, do_mc=do_mc)
        self.b5 = nn.BatchNorm2d(self.ch)
        if do_mc:
            self.mc1 = make_model({'cell': 'MultimodalController', 'input_size': self.ch, 'num_mode': num_mode,
                                   'controller_rate': config.PARAM['controller_rate']})
        self.c5 = nn.Conv2d(self.ch, img_shape[0], kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = z
        h = self.l1(h).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        if self.do_mc:
            h = self.mc1(h)
        h = nn.Tanh()(self.c5(h))
        return h


class OptimizedDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, num_mode=0, do_mc=False):
        super(OptimizedDisBlock, self).__init__()
        self.activation = nn.ReLU()
        self.do_mc = do_mc
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.c1 = nn.utils.spectral_norm(self.c1)
        self.c2 = nn.utils.spectral_norm(self.c2)
        self.c_sc = nn.utils.spectral_norm(self.c_sc)
        if do_mc:
            self.mc1 = make_model({'cell': 'MultimodalController', 'input_size': self.ch, 'num_mode': num_mode,
                                   'controller_rate': config.PARAM['controller_rate']})

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        if self.do_mc:
            h = self.mc1(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)
        return h

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1, downsample=False, num_mode=0,
                 do_mc=False):
        super(DisBlock, self).__init__()
        self.activation = nn.ReLU()
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        self.do_mc = do_mc
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c1 = nn.utils.spectral_norm(self.c1)
        self.c2 = nn.utils.spectral_norm(self.c2)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)
        if do_mc:
            self.mc1 = make_model({'cell': 'MultimodalController', 'input_size': in_channels, 'num_mode': num_mode,
                                   'controller_rate': config.PARAM['controller_rate']})
            self.mc2 = make_model({'cell': 'MultimodalController', 'input_size': hidden_channels, 'num_mode': num_mode,
                                   'controller_rate': config.PARAM['controller_rate']})

    def residual(self, x):
        h = x
        h = self.activation(h)
        if self.do_mc:
            h = self.mc1(h)
        h = self.c1(h)
        h = self.activation(h)
        if self.do_mc:
            h = self.mc2(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            if self.do_mc:
                x = self.mc1(x)
            x = self.c_sc(x)
            if self.downsample:
                return F.avg_pool2d(x, 2)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_size, num_mode=0, do_mc=False):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.ch = hidden_size
        self.activation = nn.ReLU()
        self.do_mc = do_mc
        self.block1 = OptimizedDisBlock(self.in_channels, self.ch, num_mode=num_mode, do_mc=do_mc)
        self.block2 = DisBlock(self.ch, self.ch, downsample=True, num_mode=num_mode, do_mc=do_mc)
        self.block3 = DisBlock(self.ch, self.ch, downsample=False, num_mode=num_mode, do_mc=do_mc)
        self.block4 = DisBlock(self.ch, self.ch, downsample=False, num_mode=num_mode, do_mc=do_mc)
        if do_mc:
            self.mc1 = make_model({'cell': 'MultimodalController', 'input_size': self.ch, 'num_mode': num_mode,
                                   'controller_rate': config.PARAM['controller_rate']})
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        h = h.sum(2).sum(2)
        if self.do_mc:
            h = self.mc1(h)
        output = self.l5(h)
        return output


class CGAN(nn.Module):
    def __init__(self, img_shape, latent_size, generator_hidden_size, discriminator_hidden_size, embedding_size=0,
                 num_mode=0, do_mc=False):
        super(CGAN, self).__init__()
        self.model = nn.ModuleDict({})
        self.model['generator'] = Generator(img_shape, latent_size + embedding_size, generator_hidden_size,
                                            num_mode=num_mode,
                                            do_mc=do_mc)
        self.model['discriminator'] = Discriminator(img_shape[0] + embedding_size, discriminator_hidden_size,
                                                    num_mode=num_mode,
                                                    do_mc=do_mc)
        self.model['generator_embedding'] = nn.Linear(num_mode, embedding_size, bias=False)
        self.model['discriminator_embedding'] = nn.Linear(num_mode, embedding_size, bias=False)
        self.model['discriminator_embedding'] = nn.utils.spectral_norm(self.model['discriminator_embedding'])

    def generate(self, C, x=None):
        if x is None:
            x = torch.randn([C.size(0), config.PARAM['latent_size']], device=config.PARAM['device'])
        onehot = F.one_hot(C, config.PARAM['classes_size']).float()
        embedding = self.model['generator_embedding'](onehot)
        x = torch.cat((x, embedding), dim=1)
        generated = self.model['generator'](x)
        return generated

    def discriminate(self, x, C):
        onehot = F.one_hot(C, config.PARAM['classes_size']).float()
        embedding = self.model['discriminator_embedding'](onehot)
        embedding = embedding.view([*embedding.size(), 1, 1]).expand([*embedding.size(), *x.size()[2:]])
        x = torch.cat((x, embedding), dim=1)
        discriminated = self.model['discriminator'](x)
        return discriminated

    def forward(self, input):
        x = torch.randn(input['img'].size(0), config.PARAM['latent_size'], device=config.PARAM['device'])
        x = self.generate(input['label'], x)
        x = self.discriminate(x, input['label'])
        return x


class MCGAN(nn.Module):
    def __init__(self, img_shape, latent_size, generator_hidden_size, discriminator_hidden_size, num_mode=0,
                 do_mc=False):
        super(MCGAN, self).__init__()
        self.model = nn.ModuleDict({})
        self.model['generator'] = Generator(img_shape, latent_size, generator_hidden_size, num_mode=num_mode,
                                            do_mc=do_mc)
        self.model['discriminator'] = Discriminator(img_shape[0], discriminator_hidden_size, num_mode=num_mode,
                                                    do_mc=do_mc)

    def generate(self, C, x=None):
        if x is None:
            x = torch.randn([C.size(0), config.PARAM['latent_size']], device=config.PARAM['device'])
        config.PARAM['indicator'] = F.one_hot(C, config.PARAM['classes_size']).float()
        generated = self.model['generator'](x)
        return generated

    def discriminate(self, x, C):
        config.PARAM['indicator'] = F.one_hot(C, config.PARAM['classes_size']).float()
        discriminated = self.model['discriminator'](x)
        return discriminated

    def forward(self, input):
        x = torch.randn(input['img'].size(0), config.PARAM['latent_size'], device=config.PARAM['device'])
        x = self.generate(input['label'], x)
        x = self.discriminate(x, input['label'])
        return x


def cgan():
    config.PARAM['model'] = {}
    img_shape = config.PARAM['img_shape']
    num_mode = config.PARAM['classes_size']
    latent_size = config.PARAM['latent_size']
    generator_hidden_size = config.PARAM['generator_hidden_size']
    discriminator_hidden_size = config.PARAM['discriminator_hidden_size']
    conditional_embedding_size = config.PARAM['conditional_embedding_size']
    config.PARAM['model'] = {}
    model = CGAN(img_shape, latent_size, generator_hidden_size, discriminator_hidden_size, conditional_embedding_size,
                 num_mode=num_mode, do_mc=False)
    return model


def mcgan():
    config.PARAM['model'] = {}
    img_shape = config.PARAM['img_shape']
    num_mode = config.PARAM['classes_size']
    latent_size = config.PARAM['latent_size']
    generator_hidden_size = config.PARAM['generator_hidden_size']
    discriminator_hidden_size = config.PARAM['discriminator_hidden_size']
    model = MCGAN(img_shape, latent_size, generator_hidden_size, discriminator_hidden_size, num_mode=num_mode,
                  do_mc=False)
    return model