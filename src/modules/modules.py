import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class MultimodalController(nn.Module):
    def __init__(self, input_size, num_mode):
        super().__init__()
        self.scale = 1
        self.make_codebook(input_size, num_mode)

    def make_codebook(self, input_size, num_mode):
        if cfg['mask_mode'] == 'random':
            self.register_buffer('weight', torch.rand(num_mode, input_size) * self.scale)
        elif cfg['mask_mode'] == 'learn':
            self.register_parameter('weight', nn.Parameter(torch.rand(num_mode, input_size) * self.scale))
        else:
            raise ValueError('Not valid mask mode')
        return

    def forward(self, input):
        x, label = input
        code = torch.zeros(self.weight.size(), dtype=torch.float, device=cfg['device'])
        code[self.weight > (0.5 * self.scale)] = 1
        code = (code - self.weight).detach() + self.weight
        code = code[label]
        code = code.view(*code.size(), *([1] * (x.dim() - 2)))
        x = [input[0] * code, *input[1:]]
        return x


class Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        return [self.module(input[0]), *input[1:]]
