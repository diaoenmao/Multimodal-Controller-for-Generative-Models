import config

config.init()
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import models
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, load, to_device, process_control_name, process_dataset, resume, collate, save_img
from logger import Logger


# if __name__ == "__main__":
#     test_0 = load('output/test_0.pt')
#     test_1 = load('output/test_1.pt')
#     n = 1
#     c_0 = [[] for _ in range(n)]
#     c_1 = [[] for _ in range(n)]
#     print(len(test_0), len(test_1))
#     for i in range(len(test_0)):
#         for j in range(len(test_0[i])):
#             c_0[j].append(test_0[i][j])
#     for i in range(len(test_1)):
#         for j in range(len(test_1[i])):
#             c_1[j].append(test_1[i][j])
#     for i in range(n):
#         c_0[i] = torch.cat(c_0[i], dim=0)
#         c_1[i] = torch.cat(c_1[i], dim=0)
#     for i in range(n):
#         print(torch.eq(c_0[i], c_0[i]).all())
#         print((c_0[i]-c_1[i]).abs().mean())

# if __name__ == "__main__":
#     data_name = 'Omniglot'
#     subset = 'label'
#     dataset = fetch_dataset(data_name, subset)
#     data_loader = make_data_loader(dataset)
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         print(input['img'].size())
#         print(input[subset].size())
#         break
#     save_img(input['img'], './output/img/test.png')
#     exit()

# from utils import ntuple
#
# class RLinearCell(nn.Linear):
#     def __init__(self, cell_info):
#         default_cell_info = {'bias': True, 'sharing_rate': 0}
#         cell_info = {**default_cell_info, **cell_info}
#         self.input_size = cell_info['input_size']
#         self.output_size = cell_info['output_size']
#         self.num_mode = cell_info['num_mode']
#         self.sharing_rate = cell_info['sharing_rate']
#         self.shared_size = round(self.sharing_rate * self.output_size)
#         self.free_size = self.output_size - self.shared_size
#         self.restricted_output_size = self.shared_size + self.free_size * self.num_mode
#         super(RLinearCell, self).__init__(self.input_size, self.restricted_output_size,
#                                          bias=cell_info['bias'])
#         self.register_buffer('shared_mask', torch.ones(self.shared_size))
#         # self.normalization = Normalization(cell_info['normalization'], self.output_size)
#         # self.activation = Activation(cell_info['activation'])
#
#     def forward(self, input, label):
#         mask = self.shared_mask.view(1, self.shared_mask.size(0)).expand(input.size(0), self.shared_mask.size(0))
#         mask = torch.cat((mask, label.repeat_interleave(self.free_size, dim=1).detach()), dim=1)
#         return F.linear(input, self.weight, self.bias) * mask
#
# class RConv2dCell(nn.Conv2d):
#     def __init__(self, cell_info):
#         default_cell_info = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
#                              'padding_mode': 'zeros'}
#         cell_info = {**default_cell_info, **cell_info}
#         self.input_size = cell_info['input_size']
#         self.output_size = cell_info['output_size']
#         self.num_mode = cell_info['num_mode']
#         self.sharing_rate = cell_info['sharing_rate']
#         self.shared_size = round(self.sharing_rate * self.output_size)
#         self.free_size = self.output_size - self.shared_size
#         self.restricted_output_size = self.shared_size + self.free_size * self.num_mode
#         super(RConv2dCell, self).__init__(self.input_size, self.restricted_output_size, cell_info['kernel_size'],
#                                          stride=cell_info['stride'], padding=cell_info['padding'],
#                                          dilation=cell_info['dilation'], groups=cell_info['groups'],
#                                          bias=cell_info['bias'], padding_mode=cell_info['padding_mode'])
#         self.register_buffer('shared_mask', torch.ones(self.shared_size))
#         # self.normalization = Normalization(cell_info['normalization'], self.output_size)
#         # self.activation = Activation(cell_info['activation'])
#
#     def forward(self, input, label):
#         mask = self.shared_mask.view(1, self.shared_mask.size(0)).expand(input.size(0), self.shared_mask.size(0))
#         mask = torch.cat((mask, label.repeat_interleave(self.free_size, dim=1).detach()), dim=1)
#         mask = mask.view(mask.size(0), mask.size(1), 1, 1)
#         _tuple = ntuple(2)
#         if self.padding_mode == 'circular':
#             expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2,
#                                 (self.padding[1] + 1) // 2, self.padding[1] // 2,
#                                 (self.padding[0] + 1) // 2, self.padding[0] // 2)
#             return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
#                                                                self.weight, self.bias, self.stride, _tuple(0),
#                                                                self.dilation, self.groups) * mask
#         return F.conv2d(input, self.weight, self.bias, self.stride,
#                                                            self.padding, self.dilation, self.groups) * mask
#
# if __name__ == "__main__":
#     batch_size = 10
#     num_mode = batch_size
#     input_size = 10
#     output_size = 10
#     sharing_rate = 1
#     shared_size = round(sharing_rate * output_size)
#     free_size = output_size - shared_size
#     restricted_size = shared_size + free_size * num_mode
#     cell_info_1 = {'num_mode':num_mode, 'input_size': input_size,'output_size': output_size, 'kernel_size': 1,
#     'sharing_rate':sharing_rate}
#     cell_info_2 = {'num_mode': num_mode, 'input_size': restricted_size, 'output_size': output_size, 'kernel_size':
#     1, 'sharing_rate': sharing_rate}
#     m_1 = RConv2dCell(cell_info_1)
#     m_2 = RConv2dCell(cell_info_2)
#     input = torch.randn(batch_size, input_size, 4, 4)
#     print(input.size())
#     label = torch.arange(num_mode)
#     label = label.view(label.size(0), 1)
#     onehot = label.new_zeros(label.size(0), num_mode).float()
#     onehot.scatter_(1, label, 1)
#     x = m_1(input, onehot)
#     print(x.size())
#     output = m_2(x, onehot)
#     print(output.size())
#     print(output)

# if __name__ == "__main__":
#     N = 3
#     K = 5
#     D = 6
#     input = torch.randn(N, K)
#     weight = torch.randn(K, D)
#     bias = torch.rand(D)
#     output = torch.matmul(input, weight) + bias
#     print(output)
#     N_weight = weight.view(1, K, D).expand(N, K, D)
#     N_bias = bias.view(1, D).expand(N, D)
#     N_output = input.view(N, K, 1) * N_weight
#     print(N_output.size())
#     N_output = N_output.sum(dim=1) + N_bias
#     print(N_output.size())
#     print(output-N_output)
#     print(torch.eq(output, N_output).all())

# from utils import make_restricted_output_size

class RLinearCell(nn.Linear):
    def __init__(self, cell_info):
        default_cell_info = {'bias': True, 'sharing_rate': 0}
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
        # self.normalization = Normalization(cell_info['normalization'], self.restricted_output_size)
        # self.activation = Activation(cell_info['activation'])

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
        return output


# if __name__ == "__main__":
#     batch_size = 10
#     num_mode = batch_size
#     input_size = 6
#     output_size = 5
#     sharing_rate = 0
#     shared_size = round(sharing_rate * output_size)
#     cell_info_1 = {'input_size': input_size,'output_size': output_size, 'sharing_rate': sharing_rate,
#     'num_mode':num_mode}
#     cell_info_2 = {'input_size': output_size, 'output_size': output_size, 'sharing_rate': sharing_rate, 'num_mode':
#     num_mode}
#     m_1 = RLinearCell(cell_info_1)
#     m_2 = RLinearCell(cell_info_2)
#     input = torch.randn(batch_size, input_size)
#     target = torch.randn(batch_size, output_size)
#     label = torch.arange(num_mode)
#     label = label.view(label.size(0), 1)
#     onehot = label.new_zeros(label.size(0), num_mode).float()
#     onehot.scatter_(1, label, 1)
#     config.PARAM['attr'] = onehot
#     output = m_1(input)
#     output = m_2(output)
#     loss = F.mse_loss(output, target)
#     loss.backward()


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
            x = torch.zeros(input.size(0) * self.restricted_input_size)
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
            self.running_mean = self.running_mean * exponential_average_factor + mean_i * (
                    1.0 - exponential_average_factor)
            self.running_var = self.running_var * exponential_average_factor + var_i * (
                    1.0 - exponential_average_factor)
        else:
            mean_s = torch.masked_select(self.running_mean, mask).view(input.size(0), self.input_size)
            var_s = torch.masked_select(self.running_var, mask).view(input.size(0), self.input_size)
            weight_s = torch.masked_select(self.weight, mask).view(input.size(0), self.input_size)
            bias_s = torch.masked_select(self.bias, mask).view(input.size(0), self.input_size)
            output = (input - mean_s) / torch.sqrt(var_s + self.eps) * weight_s + bias_s
        return output


if __name__ == "__main__":
    batch_size = 10
    num_mode = batch_size
    input_size = 6
    output_size = 5
    sharing_rate = 0.5
    eps = 1e-05
    input = torch.randn(batch_size, input_size)
    cell_info = {'input_size': input_size, 'sharing_rate': sharing_rate, 'num_mode': num_mode}
    m = RBatchNorm1d(cell_info)
    m.train(True)
    label = torch.arange(num_mode)
    label = label.view(label.size(0), 1)
    onehot = label.new_zeros(label.size(0), num_mode).float()
    onehot.scatter_(1, label, 1)
    config.PARAM['attr'] = onehot
    for i in range(10):
        output = m(input)
