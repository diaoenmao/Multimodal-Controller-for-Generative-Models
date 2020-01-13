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
        default_cell_info = {'bias': True, 'sharing_rate': 1}
        cell_info = {**default_cell_info, **cell_info}
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.sharing_rate = cell_info['sharing_rate']
        self.num_mode = cell_info['num_mode']
        self.shared_size = round(self.sharing_rate * self.output_size)
        self.free_size = self.output_size - self.shared_size
        self.restricted_output_size = self.shared_size + self.free_size * self.num_mode
        super(RLinearCell, self).__init__(self.input_size, self.restricted_output_size,
                                          bias=cell_info['bias'])
        self.register_buffer('shared_mask', torch.ones(self.shared_size))
        if cell_info['normalization'] == 'rbn1':
            self.normalization = RBatchNorm1d(
                {'input_size': self.output_size, 'sharing_rate': self.sharing_rate, 'num_mode': self.num_mode})
        else:
            self.normalization = Normalization(cell_info['normalization'], self.output_size)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input):
        mask = self.shared_mask.view(1, self.shared_mask.size(0)).expand(input.size(0), self.shared_mask.size(0))
        mask = torch.cat((mask, config.PARAM['attr'].repeat_interleave(self.free_size, dim=1).detach()), dim=1).bool()
        weight_mask = mask.view(mask.size(0), mask.size(1), 1)
        weight = torch.masked_select(self.weight, weight_mask).view(input.size(0), self.output_size, self.input_size)
        output = (input.view(input.size(0), 1, input.size(1)) * weight).sum(dim=2)
        if self.bias is not None:
            bias_mask = mask
            bias = torch.masked_select(self.bias, bias_mask).view(input.size(0), self.output_size)
            output = output + bias
        return self.activation(self.normalization(output))


class RBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, cell_info):
        default_cell_info = {'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True,
                             'sharing_rate': 1}
        cell_info = {**default_cell_info, **cell_info}
        self.input_size = cell_info['input_size']
        self.eps = cell_info['eps']
        self.momentum = cell_info['momentum']
        self.affine = cell_info['affine']
        self.track_running_stats = cell_info['track_running_stats']
        self.sharing_rate = cell_info['sharing_rate']
        self.num_mode = cell_info['num_mode']
        self.shared_size = round(self.sharing_rate * self.input_size)
        self.free_size = self.input_size - self.shared_size
        self.restricted_input_size = self.shared_size + self.free_size * self.num_mode
        super(RBatchNorm1d, self).__init__(self.restricted_input_size, eps=self.eps, momentum=self.momentum,
                                           affine=self.affine, track_running_stats=self.track_running_stats)
        self.register_buffer('shared_mask', torch.ones(self.shared_size))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        if self.training:
            size = input.size()
            size_prods = size[0]
            for i in range(len(size) - 2):
                size_prods *= size[i + 2]
            if size_prods == 1:
                raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
        mask = self.shared_mask.view(1, self.shared_mask.size(0)).expand(input.size(0), self.shared_mask.size(0))
        mask = torch.cat((mask, config.PARAM['attr'].repeat_interleave(self.free_size, dim=1).detach()), dim=1).bool()
        if self.training or not self.track_running_stats:
            x = input.new_zeros(input.size(0) * self.restricted_input_size)
            x[mask.view(-1)] = input.view(-1)
            x = x.view(input.size(0), self.restricted_input_size)
            n = mask.sum(dim=0)
            mean_i = x.sum(dim=0) / n
            var_i = (x.pow(2).sum(dim=0) / n - mean_i ** 2)
            var_i[n > 1] = var_i[n > 1] * (n[n > 1] / (n[n > 1] - 1))
            mean_s = torch.masked_select(mean_i, mask).view(input.size(0), self.input_size)
            var_s = torch.masked_select(var_i, mask).view(input.size(0), self.input_size)
            weight_s = torch.masked_select(self.weight, mask).view(input.size(0), self.input_size)
            bias_s = torch.masked_select(self.bias, mask).view(input.size(0), self.input_size)
            output = (input - mean_s) / torch.sqrt(var_s + self.eps) * weight_s + bias_s
            self.running_mean[n > 1] = (1.0 - exponential_average_factor) * self.running_mean[n > 1] + \
                                       exponential_average_factor * mean_i[n > 1]
            self.running_var[n > 1] = (1.0 - exponential_average_factor) * self.running_var[n > 1] + \
                                      exponential_average_factor * var_i[n > 1]
        else:
            mean_s = torch.masked_select(self.running_mean, mask).view(input.size(0), self.input_size)
            var_s = torch.masked_select(self.running_var, mask).view(input.size(0), self.input_size)
            weight_s = torch.masked_select(self.weight, mask).view(input.size(0), self.input_size)
            bias_s = torch.masked_select(self.bias, mask).view(input.size(0), self.input_size)
            output = (input - mean_s) / torch.sqrt(var_s + self.eps) * weight_s + bias_s
        return output


class RConv2dCell(nn.Conv2d):
    def __init__(self, cell_info):
        default_cell_info = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros', 'sharing_rate': 1}
        cell_info = {**default_cell_info, **cell_info}
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.sharing_rate = cell_info['sharing_rate']
        self.num_mode = cell_info['num_mode']
        self.shared_size = round(self.sharing_rate * self.output_size)
        self.free_size = self.output_size - self.shared_size
        self.restricted_output_size = self.shared_size + self.free_size * self.num_mode

        super(RConv2dCell, self).__init__(cell_info['input_size'], self.restricted_output_size,
                                          cell_info['kernel_size'], stride=cell_info['stride'],
                                          padding=cell_info['padding'], dilation=cell_info['dilation'],
                                          groups=cell_info['groups'], bias=cell_info['bias'],
                                          padding_mode=cell_info['padding_mode'])
        self.register_buffer('shared_mask', torch.ones(self.shared_size))
        if cell_info['normalization'] == 'rbn':
            self.normalization = RBatchNorm2d(
                {'input_size': self.output_size, 'sharing_rate': self.sharing_rate, 'num_mode': self.num_mode})
        else:
            self.normalization = Normalization(cell_info['normalization'], self.output_size)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input):
        mask = self.shared_mask.view(1, self.shared_mask.size(0)).expand(input.size(0), self.shared_mask.size(0))
        mask = torch.cat((mask, config.PARAM['attr'].repeat_interleave(self.free_size, dim=1).detach()), dim=1).bool()
        weight_mask = mask.view(mask.size(0), mask.size(1), 1, 1, 1)
        weight = torch.masked_select(self.weight, weight_mask).view(input.size(0), self.output_size, self.input_size,
                                                                    *self.kernel_size)
        x = F.unfold(input, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        output = (x.transpose(1, 2).unsqueeze(3) * weight.view(weight.size(0), 1, weight.size(1), -1)
                  .transpose(2, 3)).sum(2).transpose(1, 2)
        output_shape = (math.floor((input.size(2) + 2 * self.padding[0] -
                                    self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1),
                        math.floor((input.size(3) + 2 * self.padding[1] -
                                    self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        output = F.fold(output, output_shape, 1)
        if self.bias is not None:
            bias_mask = mask
            bias = torch.masked_select(self.bias, bias_mask).view(input.size(0), self.output_size, 1, 1)
            output = output + bias
        return self.activation(self.normalization(output))


class RBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, cell_info):
        default_cell_info = {'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True,
                             'sharing_rate': 1}
        cell_info = {**default_cell_info, **cell_info}
        self.input_size = cell_info['input_size']
        self.eps = cell_info['eps']
        self.momentum = cell_info['momentum']
        self.affine = cell_info['affine']
        self.track_running_stats = cell_info['track_running_stats']
        self.sharing_rate = cell_info['sharing_rate']
        self.num_mode = cell_info['num_mode']
        self.shared_size = round(self.sharing_rate * self.input_size)
        self.free_size = self.input_size - self.shared_size
        self.restricted_input_size = self.shared_size + self.free_size * self.num_mode
        super(RBatchNorm2d, self).__init__(self.restricted_input_size, eps=self.eps, momentum=self.momentum,
                                           affine=self.affine, track_running_stats=self.track_running_stats)
        self.register_buffer('shared_mask', torch.ones(self.shared_size))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        if self.training:
            size = input.size()
            size_prods = size[0]
            for i in range(len(size) - 2):
                size_prods *= size[i + 2]
            if size_prods == 1:
                raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
        mask = self.shared_mask.view(1, self.shared_mask.size(0)).expand(input.size(0), self.shared_mask.size(0))
        mask = torch.cat((mask, config.PARAM['attr'].repeat_interleave(self.free_size, dim=1).detach()), dim=1).bool()
        if self.training or not self.track_running_stats:
            x = input.new_zeros(input.size(0) * self.restricted_input_size * input.size(2) * input.size(3))
            x[mask.view(*mask.size(), 1, 1).expand(
                mask.size(0), mask.size(1), input.size(2), input.size(3)).reshape(-1)] = input.view(-1)
            x = x.view(input.size(0), self.restricted_input_size, input.size(2), input.size(3))
            n = mask.sum(dim=0) * input.size(2) * input.size(3)
            mean_i = x.sum(dim=(0, 2, 3)) / n
            var_i = (x.pow(2).sum(dim=(0, 2, 3)) / n - mean_i ** 2)
            var_i[n > 1] = var_i[n > 1] * (n[n > 1] / (n[n > 1] - 1))
            mean_s = torch.masked_select(mean_i, mask).view(input.size(0), self.input_size, 1, 1)
            var_s = torch.masked_select(var_i, mask).view(input.size(0), self.input_size, 1, 1)
            weight_s = torch.masked_select(self.weight, mask).view(input.size(0), self.input_size, 1, 1)
            bias_s = torch.masked_select(self.bias, mask).view(input.size(0), self.input_size, 1, 1)
            output = (input - mean_s) / torch.sqrt(var_s + self.eps) * weight_s + bias_s
            self.running_mean[n > 1] = (1.0 - exponential_average_factor) * self.running_mean[n > 1] + \
                                       exponential_average_factor * mean_i[n > 1]
            self.running_var[n > 1] = (1.0 - exponential_average_factor) * self.running_var[n > 1] + \
                                      exponential_average_factor * var_i[n > 1]
        else:
            mean_s = torch.masked_select(self.running_mean, mask).view(input.size(0), self.input_size, 1, 1)
            var_s = torch.masked_select(self.running_var, mask).view(input.size(0), self.input_size, 1, 1)
            weight_s = torch.masked_select(self.weight, mask).view(input.size(0), self.input_size, 1, 1)
            bias_s = torch.masked_select(self.bias, mask).view(input.size(0), self.input_size, 1, 1)
            output = (input - mean_s) / torch.sqrt(var_s + self.eps) * weight_s + bias_s
        return output