import config
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import make_model


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, do_mc=False, num_mode=None,
                 controller_rate=None):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        if do_mc:
            self.mc_1 = make_model({'cell': 'MultimodalController', 'input_size': in_channels, 'num_mode': num_mode,
                                    'controller_rate': controller_rate})
            self.mc_2 = make_model({'cell': 'MultimodalController', 'input_size': out_channels, 'num_mode': num_mode,
                                    'controller_rate': controller_rate})
            self.model = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                self.mc_1,
                nn.Upsample(scale_factor=2),
                self.conv1,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                self.mc_2,
                self.conv2
            )
            if in_channels != out_channels or stride != 1:
                self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
                nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
                if stride != 1:
                    self.bypass = nn.Sequential(
                        self.mc_1,
                        nn.Upsample(scale_factor=2),
                        self.bypass_conv,
                    )
                else:
                    self.bypass = nn.Sequential(
                        self.mc_1,
                        self.bypass_conv,
                    )
            else:
                self.bypass = self.mc_1
        else:
            self.model = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                self.conv1,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                self.conv2
            )
            if in_channels != out_channels or stride != 1:
                self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
                nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
                if stride != 1:
                    self.bypass = nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        self.bypass_conv,
                    )
                else:
                    self.bypass = self.bypass_conv
            else:
                self.bypass = nn.Identity()

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, do_mc=False, num_mode=None,
                 controller_rate=None):
        super(ResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        if do_mc:
            self.mc_1 = make_model({'cell': 'MultimodalController', 'input_size': in_channels, 'num_mode': num_mode,
                                    'controller_rate': controller_rate})
            self.mc_2 = make_model({'cell': 'MultimodalController', 'input_size': out_channels, 'num_mode': num_mode,
                                    'controller_rate': controller_rate})
            if stride == 1:
                self.model = nn.Sequential(
                    nn.ReLU(),
                    self.mc_1,
                    torch.nn.utils.spectral_norm(self.conv1),
                    nn.ReLU(),
                    self.mc_2,
                    torch.nn.utils.spectral_norm(self.conv2)
                )
            else:
                self.model = nn.Sequential(
                    nn.ReLU(),
                    self.mc_1,
                    torch.nn.utils.spectral_norm(self.conv1),
                    nn.ReLU(),
                    self.mc_2,
                    torch.nn.utils.spectral_norm(self.conv2),
                    nn.AvgPool2d(2, stride=stride, padding=0)
                )
            if in_channels != out_channels or stride != 1:
                self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
                nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
                if stride != 1:
                    self.bypass = nn.Sequential(
                        self.mc_1,
                        torch.nn.utils.spectral_norm(self.bypass_conv),
                        nn.AvgPool2d(2, stride=stride, padding=0)
                    )
                else:
                    self.bypass = nn.Sequential(
                        self.mc_1,
                        torch.nn.utils.spectral_norm(self.bypass_conv)
                    )
            else:
                self.bypass = self.mc_1
        else:
            if stride == 1:
                self.model = nn.Sequential(
                    nn.ReLU(),
                    torch.nn.utils.spectral_norm(self.conv1),
                    nn.ReLU(),
                    torch.nn.utils.spectral_norm(self.conv2)
                )
            else:
                self.model = nn.Sequential(
                    nn.ReLU(),
                    torch.nn.utils.spectral_norm(self.conv1),
                    nn.ReLU(),
                    torch.nn.utils.spectral_norm(self.conv2),
                    nn.AvgPool2d(2, stride=stride, padding=0)
                )
            if in_channels != out_channels or stride != 1:
                self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
                nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
                if stride != 1:
                    self.bypass = nn.Sequential(
                        torch.nn.utils.spectral_norm(self.bypass_conv),
                        nn.AvgPool2d(2, stride=stride, padding=0)
                    )
                else:
                    self.bypass = nn.Sequential(
                        torch.nn.utils.spectral_norm(self.bypass_conv)
                    )
            else:
                self.bypass = nn.Identity()

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, do_mc=False, num_mode=None,
                 controller_rate=None):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        if do_mc:
            # we don't want to apply ReLU activation to raw image before convolution transformation.
            self.mc_1 = make_model({'cell': 'MultimodalController', 'input_size': out_channels, 'num_mode': num_mode,
                                    'controller_rate': controller_rate})
            self.model = nn.Sequential(
                torch.nn.utils.spectral_norm(self.conv1),
                nn.ReLU(),
                self.mc_1,
                torch.nn.utils.spectral_norm(self.conv2),
                nn.AvgPool2d(2)
            )
            self.bypass = nn.Sequential(
                torch.nn.utils.spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2),
            )
        else:
            self.model = nn.Sequential(
                torch.nn.utils.spectral_norm(self.conv1),
                nn.ReLU(),
                torch.nn.utils.spectral_norm(self.conv2),
                nn.AvgPool2d(2)
            )
            self.bypass = nn.Sequential(
                torch.nn.utils.spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2),
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Generator(nn.Module):
    def __init__(self, in_channels, latent_size, hidden_size, stride, do_mc=False, conditional_embedding_size=None,
                 num_mode=None, controller_rate=None):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.do_mc = do_mc
        if do_mc:
            self.mc = make_model({'cell': 'MultimodalController', 'input_size': hidden_size[-1], 'num_mode': num_mode,
                                  'controller_rate': controller_rate})
            self.dense = nn.Linear(self.latent_size, 4 * 4 * hidden_size[0])

        else:
            self.embedding = nn.Linear(num_mode, conditional_embedding_size, bias=False)
            self.dense = nn.Linear(self.latent_size + conditional_embedding_size, 4 * 4 * hidden_size[0])

        self.final = nn.Conv2d(hidden_size[-1], in_channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)
        if do_mc:
            self.model = []
            for i in range(1, len(hidden_size)):
                self.model.append(
                    ResBlockGenerator(hidden_size[i - 1], hidden_size[i], stride=stride[i - 1], do_mc=do_mc,
                                      num_mode=num_mode, controller_rate=controller_rate))
            self.model.extend([nn.BatchNorm2d(hidden_size[-1]), nn.ReLU(), self.mc, self.final, nn.Tanh()])
        else:
            self.model = []
            for i in range(1, len(hidden_size)):
                self.model.append(ResBlockGenerator(hidden_size[i - 1], hidden_size[i], stride=stride[i - 1]))
            self.model.extend([nn.BatchNorm2d(hidden_size[-1]), nn.ReLU(), self.final, nn.Tanh()])
        self.model = nn.Sequential(*self.model)

    def forward(self, z, label):
        config.PARAM['indicator'] = F.one_hot(label, config.PARAM['classes_size']).float()
        if not self.do_mc:
            embedding = self.embedding(config.PARAM['indicator'])
            z = torch.cat([z, embedding], dim=1)
        return self.model(self.dense(z).view(z.size(0), -1, 4, 4))


class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_size, stride, do_mc=False, conditional_embedding_size=None, num_mode=None,
                 controller_rate=None):
        super(Discriminator, self).__init__()
        self.do_mc = do_mc
        self.model = []
        if do_mc:
            self.mc = make_model({'cell': 'MultimodalController', 'input_size': hidden_size[-1], 'num_mode': num_mode,
                                  'controller_rate': controller_rate})
            self.model = []
            for i in range(len(hidden_size)):
                if i == 0:
                    self.model.append(
                        FirstResBlockDiscriminator(in_channels, hidden_size[i], do_mc=do_mc, num_mode=num_mode,
                                                   controller_rate=controller_rate))
                else:
                    self.model.append(
                        ResBlockDiscriminator(hidden_size[i - 1], hidden_size[i], stride=stride[i], do_mc=do_mc,
                                              num_mode=num_mode, controller_rate=controller_rate))
            self.model.extend([nn.ReLU(), self.mc, nn.AdaptiveAvgPool2d(1)])
        else:
            self.embedding = nn.Linear(num_mode, conditional_embedding_size, bias=False)
            self.embedding = torch.nn.utils.spectral_norm(self.embedding)
            self.model = []
            for i in range(len(hidden_size)):
                if i == 0:
                    self.model.append(
                        FirstResBlockDiscriminator(in_channels + conditional_embedding_size, hidden_size[i]))
                else:
                    self.model.append(
                        ResBlockDiscriminator(hidden_size[i - 1], hidden_size[i], stride=stride[i]))
            self.model.extend([nn.ReLU(), nn.AdaptiveAvgPool2d(1)])
        self.model = nn.Sequential(*self.model)
        self.fc = nn.Linear(hidden_size[-1], 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = torch.nn.utils.spectral_norm(self.fc)

    def forward(self, x, label):
        config.PARAM['indicator'] = F.one_hot(label, config.PARAM['classes_size']).float()
        if not self.do_mc:
            embedding = self.embedding(config.PARAM['indicator'])
            embedding = embedding.view([*embedding.size(), 1, 1]).expand([*embedding.size(), *x.size()[2:]])
            x = torch.cat([x, embedding], dim=1)
        return self.fc(self.model(x).view(x.size(0), -1))


def cgan():
    img_shape = config.PARAM['img_shape']
    num_mode = config.PARAM['classes_size']
    latent_size = config.PARAM['latent_size']
    conditional_embedding_size = config.PARAM['conditional_embedding_size']
    generator_hidden_size = config.PARAM['generator_hidden_size']
    discriminator_hidden_size = config.PARAM['discriminator_hidden_size']
    generator_stride = config.PARAM['generator_stride']
    discriminator_stride = config.PARAM['discriminator_stride']
    generator = Generator(img_shape[0], latent_size, generator_hidden_size, generator_stride, do_mc=False,
                          conditional_embedding_size=conditional_embedding_size, num_mode=num_mode)
    discriminator = Discriminator(img_shape[0], discriminator_hidden_size, discriminator_stride, do_mc=False,
                                  conditional_embedding_size=conditional_embedding_size, num_mode=num_mode)
    model = nn.ModuleDict({'generator': generator, 'discriminator': discriminator})
    return model


def mcgan():
    img_shape = config.PARAM['img_shape']
    num_mode = config.PARAM['classes_size']
    controller_rate = config.PARAM['controller_rate']
    latent_size = config.PARAM['latent_size']
    generator_hidden_size = config.PARAM['generator_hidden_size']
    discriminator_hidden_size = config.PARAM['discriminator_hidden_size']
    generator_stride = config.PARAM['generator_stride']
    discriminator_stride = config.PARAM['discriminator_stride']
    generator = Generator(img_shape[0], latent_size, generator_hidden_size, generator_stride, do_mc=True,
                          conditional_embedding_size=None, num_mode=num_mode, controller_rate=controller_rate)
    discriminator = Discriminator(img_shape[0], discriminator_hidden_size, discriminator_stride, do_mc=True,
                                  conditional_embedding_size=None, num_mode=num_mode, controller_rate=controller_rate)
    model = nn.ModuleDict({'generator': generator, 'discriminator': discriminator})
    return model