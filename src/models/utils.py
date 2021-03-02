import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import cfg


def init_param(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    return m


def compute_gradient_penalty(discriminator, real, fake, label):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=cfg['device'])
    interpolate = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
    interpolate_validity = discriminator(interpolate, label)
    fake = torch.ones(real.shape[0], 1, device=cfg['device'])
    gradients = torch.autograd.grad(outputs=interpolate_validity, inputs=interpolate, grad_outputs=fake,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def discriminator_loss_fn(real_validity, fake_validity, discriminator, real, fake, label):
    lambda_gp = 10
    gradient_penalty = compute_gradient_penalty(discriminator, real, fake, label)
    D_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
    return D_loss


def generator_loss_fn(fake_validity):
    G_loss = -torch.mean(fake_validity)
    return G_loss


def classifier_loss_fn(output, target):
    classifier_loss = F.cross_entropy(output, target)
    return classifier_loss
