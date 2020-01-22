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
    elif cell_info['cell'] == 'RLinearCell':
        cell = RLinearCell(cell_info)
    elif cell_info['cell'] == 'RConv2dCell':
        cell = RConv2dCell(cell_info)
    elif cell_info['cell'] == 'RConvTranspose2dCell':
        cell = RConvTranspose2dCell(cell_info)
    elif cell_info['cell'] == 'ResRConv2dCell':
        cell = ResRConv2dCell(cell_info)
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
        conv2_info = {**cell_info, 'normalization': 'none', 'activation': 'none'}
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.conv1 = Conv2dCell(conv1_info)
        self.conv2 = Conv2dCell(conv2_info)
        self.normalization = Normalization(cell_info['normalization'], self.output_size)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input):
        identity = input
        x = self.conv1(input)
        x = self.conv2(x)
        output = self.activation(x + identity)
        return output


class RLinearCell(nn.Linear):
    def __init__(self, cell_info):
        default_cell_info = {'bias': True, 'sharing_rate': 1, 'num_mode': 1}
        cell_info = {**default_cell_info, **cell_info}
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.num_mode = cell_info['num_mode']
        super(RLinearCell, self).__init__(cell_info['input_size'], cell_info['output_size'],
                                          bias=cell_info['bias'])
        self.embedding = nn.Linear(self.num_mode, self.output_size, bias=False)
        self.normalization = Normalization(cell_info['normalization'], self.output_size)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input):
        embedding = self.embedding(config.PARAM['attr'])
        p = torch.sigmoid(embedding)
        mask = (p.round() - p).detach() + p
        output = F.linear(input, self.weight, self.bias)
        output = output * mask
        return self.activation(self.normalization(output))


class RConv2dCell(nn.Conv2d):
    def __init__(self, cell_info):
        default_cell_info = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros', 'sharing_rate': 1, 'num_mode': 1}
        cell_info = {**default_cell_info, **cell_info}
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.num_mode = cell_info['num_mode']
        super(RConv2dCell, self).__init__(cell_info['input_size'], cell_info['output_size'],
                                          cell_info['kernel_size'], stride=cell_info['stride'],
                                          padding=cell_info['padding'], dilation=cell_info['dilation'],
                                          groups=cell_info['groups'], bias=cell_info['bias'],
                                          padding_mode=cell_info['padding_mode'])
        self.embedding = nn.Linear(self.num_mode, self.output_size, bias=False)
        self.normalization = Normalization(cell_info['normalization'], self.output_size)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input):
        _tuple = ntuple(2)
        embedding = self.embedding(config.PARAM['attr'])
        p = torch.sigmoid(embedding)
        mask = (p.round() - p).detach() + p
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv2d(F.pad(input, expanded_padding, mode='circular'), self.weight, self.bias, self.stride,
                              _tuple(0), self.dilation, self.groups)
        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output = output * mask.view(*mask.size(), 1, 1)
        return self.activation(self.normalization(output))


class RConvTranspose2dCell(nn.ConvTranspose2d):
    def __init__(self, cell_info):
        default_cell_info = {'stride': 1, 'padding': 0, 'output_padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros', 'sharing_rate': 1, 'num_mode': 1}
        cell_info = {**default_cell_info, **cell_info}
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.sharing_rate = cell_info['sharing_rate']
        self.num_mode = cell_info['num_mode']
        super(RConvTranspose2dCell, self).__init__(cell_info['input_size'], cell_info['output_size'],
                                                   cell_info['kernel_size'],
                                                   stride=cell_info['stride'], padding=cell_info['padding'],
                                                   output_padding=cell_info['output_padding'],
                                                   dilation=cell_info['dilation'], groups=cell_info['groups'],
                                                   bias=cell_info['bias'], padding_mode=cell_info['padding_mode'])
        self.embedding = nn.Linear(self.num_mode, self.output_size, bias=False)
        self.normalization = Normalization(cell_info['normalization'], self.output_size)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        embedding = self.embedding(config.PARAM['attr'])
        p = torch.sigmoid(embedding)
        mask = (p.round() - p).detach() + p
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        output = F.conv_transpose2d(input, self.weight, self.bias, self.stride, self.padding, output_padding,
                                    self.groups, self.dilation)
        output = output * mask.view(*mask.size(), 1, 1)
        return self.activation(self.normalization(output))


class ResRConv2dCell(nn.Module):
    def __init__(self, cell_info):
        super(ResRConv2dCell, self).__init__()
        default_cell_info = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros', 'sharing_rate': 1, 'num_mode': 1}
        cell_info = {**default_cell_info, **cell_info}
        conv1_info = {**cell_info}
        conv2_info = {**cell_info, 'normalization': 'none', 'activation': 'none'}
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.conv1 = RConv2dCell(conv1_info)
        self.conv2 = RConv2dCell(conv2_info)
        self.normalization = Normalization(cell_info['normalization'], self.output_size)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input):
        identity = input
        x = self.conv1(input)
        x = self.conv2(x)
        output = self.activation(x + identity)
        return output