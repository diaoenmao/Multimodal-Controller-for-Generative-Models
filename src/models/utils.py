import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import cfg


def init_param(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    if cfg['model_name'] in ['mcgan']:
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight.data, 1.)
    return m


def make_SpectralNormalization(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        return torch.nn.utils.spectral_norm(m)
    else:
        return m


def discriminator_loss_fn(real_validity, fake_validity):
    D_loss = F.relu(1.0 - real_validity).mean() + F.relu(1.0 + fake_validity).mean()
    return D_loss


def generator_loss_fn(fake_validity):
    G_loss = -fake_validity.mean()
    return G_loss


def classifier_loss_fn(output, target):
    classifier_loss = F.cross_entropy(output, target)
    return classifier_loss
