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
    elif cell_info['cell'] == 'Resize':
        cell = ResizeCell(cell_info)
    elif cell_info['cell'] == 'LinearCell':
        cell = LinearCell(cell_info)
    elif cell_info['cell'] == 'Conv2dCell':
        cell = Conv2dCell(cell_info)
    elif cell_info['cell'] == 'ConvTranspose2dCell':
        cell = ConvTranspose2dCell(cell_info)
    elif cell_info['cell'] == 'ResConv2dCell':
        cell = ResConv2dCell(cell_info)
    elif cell_info['cell'] == 'MCLinearCell':
        cell = MCLinearCell(cell_info)
    elif cell_info['cell'] == 'MCConv2dCell':
        cell = MCConv2dCell(cell_info)
    elif cell_info['cell'] == 'MCConvTranspose2dCell':
        cell = MCConvTranspose2dCell(cell_info)
    elif cell_info['cell'] == 'MCResConv2dCell':
        cell = MCResConv2dCell(cell_info)
    elif cell_info['cell'] == 'MultimodalController':
        cell = MultimodalController(cell_info)
    elif cell_info['cell'] == 'VectorQuantization':
        cell = VectorQuantization(cell_info)
    else:
        raise ValueError('Not valid cell info: {}'.format(cell_info))
    return cell


def Normalization(mode, size, dim=2):
    if mode == 'none':
        return nn.Identity()
    elif mode == 'bn':
        if dim == 1:
            return nn.BatchNorm1d(size)
        elif dim == 2:
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


class ResizeCell(nn.Module):
    def __init__(self, cell_info):
        super(ResizeCell, self).__init__()
        self.resize = cell_info['resize']

    def forward(self, input):
        return input.view(input.size(0), *self.resize)


class LinearCell(nn.Linear):
    def __init__(self, cell_info):
        default_cell_info = {'bias': True}
        cell_info = {**default_cell_info, **cell_info}
        super(LinearCell, self).__init__(cell_info['input_size'], cell_info['output_size'], bias=cell_info['bias'])
        self.normalization = Normalization(cell_info['normalization'], cell_info['output_size'], 1)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input):
        return self.activation(self.normalization(F.linear(input, self.weight, self.bias)))


class MCLinearCell(nn.Linear):
    def __init__(self, cell_info):
        default_cell_info = {'bias': True}
        cell_info = {**default_cell_info, **cell_info}
        super(MCLinearCell, self).__init__(cell_info['input_size'], cell_info['output_size'], bias=cell_info['bias'])
        self.normalization = Normalization(cell_info['normalization'], self.output_size, 1)
        self.activation = Activation(cell_info['activation'])
        self.mc = MultimodalController(
            {'cell': 'MultimodalController', 'input_size': cell_info['output_size'], 'num_mode': cell_info['num_mode'],
             'controller_rate': cell_info['controller_rate']})

    def forward(self, input):
        return self.mc(self.activation(self.normalization(F.linear(input, self.weight, self.bias))))


class Conv2dCell(nn.Conv2d):
    def __init__(self, cell_info):
        default_cell_info = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros'}
        cell_info = {**default_cell_info, **cell_info}
        super(Conv2dCell, self).__init__(cell_info['input_size'], cell_info['output_size'], cell_info['kernel_size'],
                                         stride=cell_info['stride'], padding=cell_info['padding'],
                                         dilation=cell_info['dilation'], groups=cell_info['groups'],
                                         bias=cell_info['bias'], padding_mode=cell_info['padding_mode'])
        self.normalization = Normalization(cell_info['normalization'], cell_info['output_size'])
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


class MCConv2dCell(nn.Conv2d):
    def __init__(self, cell_info):
        default_cell_info = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros'}
        cell_info = {**default_cell_info, **cell_info}
        super(MCConv2dCell, self).__init__(cell_info['input_size'], cell_info['output_size'], cell_info['kernel_size'],
                                           stride=cell_info['stride'], padding=cell_info['padding'],
                                           dilation=cell_info['dilation'], groups=cell_info['groups'],
                                           bias=cell_info['bias'], padding_mode=cell_info['padding_mode'])
        self.normalization = Normalization(cell_info['normalization'], cell_info['output_size'])
        self.activation = Activation(cell_info['activation'])
        self.mc = MultimodalController(
            {'cell': 'MultimodalController', 'input_size': cell_info['output_size'], 'num_mode': cell_info['num_mode'],
             'controller_rate': cell_info['controller_rate']})

    def forward(self, input):
        _tuple = ntuple(2)
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2,
                                (self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return self.mc(self.activation(self.normalization(F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                                                       self.weight, self.bias, self.stride, _tuple(0),
                                                                       self.dilation, self.groups))))
        return self.mc(self.activation(self.normalization(F.conv2d(input, self.weight, self.bias, self.stride,
                                                                   self.padding, self.dilation, self.groups))))


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
        self.normalization = Normalization(cell_info['normalization'], cell_info['output_size'])
        self.activation = Activation(cell_info['activation'])

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return self.activation(self.normalization(F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)))


class MCConvTranspose2dCell(nn.ConvTranspose2d):
    def __init__(self, cell_info):
        default_cell_info = {'stride': 1, 'padding': 0, 'output_padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros'}
        cell_info = {**default_cell_info, **cell_info}
        super(MCConvTranspose2dCell, self).__init__(cell_info['input_size'], cell_info['output_size'],
                                                    cell_info['kernel_size'],
                                                    stride=cell_info['stride'], padding=cell_info['padding'],
                                                    output_padding=cell_info['output_padding'],
                                                    dilation=cell_info['dilation'], groups=cell_info['groups'],
                                                    bias=cell_info['bias'], padding_mode=cell_info['padding_mode'])
        self.normalization = Normalization(cell_info['normalization'], cell_info['output_size'])
        self.activation = Activation(cell_info['activation'])
        self.mc = MultimodalController(
            {'cell': 'MultimodalController', 'input_size': cell_info['output_size'], 'num_mode': cell_info['num_mode'],
             'controller_rate': cell_info['controller_rate']})

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return self.mc(self.activation(self.normalization(F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation))))


class ResConv2dCell(nn.Module):
    def __init__(self, cell_info):
        super(ResConv2dCell, self).__init__()
        default_cell_info = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros'}
        cell_info = {**default_cell_info, **cell_info}
        conv1_info = {**cell_info, 'output_size': cell_info['res_size']}
        conv2_info = {**cell_info, 'input_size': cell_info['res_size'], 'kernel_size': 1, 'stride': 1, 'padding': 0,
                      'activation': 'none'}
        self.conv1 = Conv2dCell(conv1_info)
        self.conv2 = Conv2dCell(conv2_info)
        if cell_info['stride'] > 1 or cell_info['input_size'] != cell_info['output_size']:
            self.shortcut = Conv2dCell({**cell_info, 'kernel_size': 1, 'stride': 1, 'padding': 0,
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


class MCResConv2dCell(nn.Module):
    def __init__(self, cell_info):
        super(MCResConv2dCell, self).__init__()
        default_cell_info = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros'}
        cell_info = {**default_cell_info, **cell_info}
        conv1_info = {**cell_info, 'output_size': cell_info['res_size']}
        conv2_info = {**cell_info, 'input_size': cell_info['res_size'], 'kernel_size': 1, 'stride': 1, 'padding': 0,
                      'activation': 'none'}
        self.conv1 = MCConv2dCell(conv1_info)
        self.conv2 = MCConv2dCell(conv2_info)
        if cell_info['stride'] > 1 or cell_info['input_size'] != cell_info['output_size']:
            self.shortcut = MCConv2dCell({**cell_info, 'kernel_size': 1, 'stride': 1, 'padding': 0,
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


class VectorQuantization(nn.Module):
    def __init__(self, cell_info):
        default_cell_info = {'decay': 0.99, 'eps': 1e-5}
        cell_info = {**default_cell_info, **cell_info}
        super(VectorQuantization, self).__init__()
        self.embedding_dim = cell_info['embedding_dim']
        self.num_embedding = cell_info['num_embedding']
        self.decay = cell_info['decay']
        self.eps = cell_info['eps']
        embedding = torch.randn(self.embedding_dim, self.num_embedding)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(self.num_embedding))
        self.register_buffer('embedding_avg', embedding.clone())

    def forward(self, input):
        input = input.permute(0, 2, 3, 1).contiguous()
        input_shape = input.size()
        flatten = input.view(-1, self.embedding_dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embedding
                + self.embedding.pow(2).sum(0, keepdim=True)
        )
        _, embedding_ind = dist.min(1)
        embedding_onehot = F.one_hot(embedding_ind, self.num_embedding).type(flatten.dtype)
        embedding_ind = embedding_ind.view(*input_shape[:-1])
        quantize = self.embedding_code(embedding_ind)
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embedding_onehot.sum(0)
            )
            embedding_sum = flatten.transpose(0, 1) @ embedding_onehot
            self.embedding_avg.data.mul_(self.decay).add_(1 - self.decay, embedding_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.num_embedding * self.eps) * n
            )
            embedding_normalized = self.embedding_avg / cluster_size.unsqueeze(0)
            self.embedding.data.copy_(embedding_normalized)
        diff = F.mse_loss(quantize.detach(), input)
        quantize = input + (quantize - input).detach()
        quantize = quantize.permute(0, 3, 1, 2).contiguous()
        return quantize, diff, embedding_ind

    def embedding_code(self, embedding_ind):
        return F.embedding(embedding_ind, self.embedding.transpose(0, 1))


class MultimodalController(nn.Module):
    def __init__(self, cell_info):
        super(MultimodalController, self).__init__()
        default_cell_info = {'num_mode': 1, 'controller_rate': 1}
        cell_info = {**default_cell_info, **cell_info}
        self.input_size = cell_info['input_size']
        self.num_mode = cell_info['num_mode']
        self.controller_rate = cell_info['controller_rate']
        codebook = self.make_codebook()
        self.register_buffer('codebook', codebook)

    def make_codebook(self):
        if self.controller_rate == 1:
            codebook = torch.ones(self.num_mode, self.input_size, dtype=torch.float)
        else:
            d = torch.distributions.bernoulli.Bernoulli(probs=self.controller_rate)
            codebook = set()
            while len(codebook) < self.num_mode:
                codebook_c = d.sample((self.num_mode, self.input_size))
                codebook_c = [tuple(c) for c in codebook_c.tolist()]
                codebook.update(codebook_c)
            codebook = torch.tensor(list(codebook)[:self.num_mode], dtype=torch.float)
        return codebook

    def forward(self, input):
        code = config.PARAM['indicator'].matmul(self.codebook)
        code = code.view(*code.size(), *([1] * (input.dim() - 2)))
        output = input * code.detach()
        return output