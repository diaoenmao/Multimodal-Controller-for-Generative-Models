import config
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ntuple


def make_cell(cell_info):
    if cell_info['cell'] == 'none':
        cell = nn.Identity()
    elif cell_info['cell'] == 'Normalization':
        cell = Normalization(cell_info)
    elif cell_info['cell'] == 'Activation':
        cell = Activation(cell_info)
    elif cell_info['cell'] == 'LinearCell':
        cell = LinearCell(cell_info)
    elif cell_info['cell'] == 'Conv2dCell':
        cell = Conv2dCell(cell_info)
    elif cell_info['cell'] == 'ConvTranspose2dCell':
        cell = ConvTranspose2dCell(cell_info)
    elif cell_info['cell'] == 'ResConv2dCell':
        cell = ResConv2dCell(cell_info)
    elif cell_info['cell'] == 'MultimodalController':
        cell = MultimodalController(cell_info)
    else:
        raise ValueError('Not valid cell info: {}'.format(cell_info))
    return cell


def Normalization(mode, size):
    if mode == 'none':
        return nn.Identity()
    elif mode == 'bn1':
        return nn.BatchNorm1d(size)
    elif mode == 'bn':
        return nn.BatchNorm2d(size)
    elif mode == 'in':
        return nn.InstanceNorm2d(size)
    elif mode == 'ln':
        return nn.LayerNorm(size)
    else:
        raise ValueError('Not valid normalization')
    return


def Activation(mode):
    if mode == 'none':
        return nn.Sequential()
    elif mode == 'tanh':
        return nn.Tanh()
    elif mode == 'hardtanh':
        return nn.Hardtanh()
    elif mode == 'relu':
        return nn.ReLU(inplace=True)
    elif mode == 'prelu':
        return nn.PReLU()
    elif mode == 'elu':
        return nn.ELU(inplace=True)
    elif mode == 'selu':
        return nn.SELU(inplace=True)
    elif mode == 'celu':
        return nn.CELU(inplace=True)
    elif mode == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=False)
    elif mode == 'sigmoid':
        return nn.Sigmoid()
    elif mode == 'softmax':
        return nn.SoftMax()
    else:
        raise ValueError('Not valid activation')
    return


class LinearCell(nn.Linear):
    def __init__(self, cell_info):
        default_cell_info = {'bias': True}
        cell_info = {**default_cell_info, **cell_info}
        super(LinearCell, self).__init__(cell_info['input_size'], cell_info['output_size'], bias=cell_info['bias'])
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.normalization = Normalization(cell_info['normalization'], self.output_size)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input):
        return self.activation(self.normalization(F.linear(input, self.weight, self.bias)))


class Conv2dCell(nn.Conv2d):
    def __init__(self, cell_info):
        default_cell_info = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros'}
        cell_info = {**default_cell_info, **cell_info}
        super(Conv2dCell, self).__init__(cell_info['input_size'], cell_info['output_size'], cell_info['kernel_size'],
                                         stride=cell_info['stride'], padding=cell_info['padding'],
                                         dilation=cell_info['dilation'], groups=cell_info['groups'],
                                         bias=cell_info['bias'], padding_mode=cell_info['padding_mode'])
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.normalization = Normalization(cell_info['normalization'], self.output_size)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input):
        _tuple = ntuple(2)
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2,
                                (self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return self.activation(self.normalization(F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                                               self.weight, self.bias, self.stride, _tuple(0),
                                                               self.dilation, self.groups)))
        return self.activation(self.normalization(F.conv2d(input, self.weight, self.bias, self.stride,
                                                           self.padding, self.dilation, self.groups)))


class ConvTranspose2dCell(nn.ConvTranspose2d):
    def __init__(self, cell_info):
        default_cell_info = {'stride': 1, 'padding': 0, 'output_padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros'}
        cell_info = {**default_cell_info, **cell_info}
        super(ConvTranspose2dCell, self).__init__(cell_info['input_size'], cell_info['output_size'],
                                                  cell_info['kernel_size'],
                                                  stride=cell_info['stride'], padding=cell_info['padding'],
                                                  output_padding=cell_info['output_padding'],
                                                  dilation=cell_info['dilation'], groups=cell_info['groups'],
                                                  bias=cell_info['bias'], padding_mode=cell_info['padding_mode'])
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.normalization = Normalization(cell_info['normalization'], self.output_size)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return self.activation(self.normalization(F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)))


class ResConv2dCell(nn.Module):
    def __init__(self, cell_info):
        super(ResConv2dCell, self).__init__()
        default_cell_info = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros'}
        cell_info = {**default_cell_info, **cell_info}
        conv1_info = {**cell_info}
        conv2_info = {**cell_info, 'input_size': cell_info['output_size'], 'stride': 1,
                      'normalization': cell_info['normalization'], 'activation': 'none'}
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.conv1 = Conv2dCell(conv1_info)
        self.conv2 = Conv2dCell(conv2_info)
        if cell_info['stride'] > 1 or cell_info['input_size'] != cell_info['output_size']:
            self.shortcut = Conv2dCell({**cell_info, 'kernel_size': 1, 'padding': 0,
                                        'normalization': cell_info['normalization'], 'activation': 'none'})
        else:
            self.shortcut = nn.Identity()
        self.activation = Activation(cell_info['activation'])

    def forward(self, input):
        shortcut = self.shortcut(input)
        x = self.conv1(input)
        x = self.conv2(x)
        output = self.activation(x + shortcut)
        return output


class MultimodalController(nn.Module):
    def __init__(self, cell_info):
        super(MultimodalController, self).__init__()
        default_cell_info = {'sharing_rate': 1, 'num_mode': 1}
        cell_info = {**default_cell_info, **cell_info}
        self.input_size = cell_info['input_size']
        self.sharing_rate = cell_info['sharing_rate']
        self.num_mode = cell_info['num_mode']
        self.mode_size = math.ceil(self.input_size * (1 - self.sharing_rate) / self.num_mode)
        self.free_size = self.mode_size * self.num_mode
        self.shared_size = self.input_size - self.free_size
        embedding = torch.zeros(self.num_mode, self.input_size)
        if self.shared_size > 0:
            embedding[:, :self.shared_size] = 1
        if self.free_size > 0:
            idx = torch.arange(self.num_mode).repeat_interleave(self.mode_size, dim=0).view(1, -1)
            embedding[:, self.shared_size:].scatter_(0, idx, 1)
        self.register_buffer('embedding', embedding)

    def forward(self, input):
        embedding = config.PARAM['attr'].matmul(self.embedding)
        embedding = embedding.view(*embedding.size(), *([1] * (input.dim() - 2)))
        output = input * embedding
        return output