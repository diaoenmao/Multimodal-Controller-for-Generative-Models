import config

config.init()
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import math
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

# class RLinearCell(nn.Linear):
#     def __init__(self, cell_info):
#         default_cell_info = {'bias': True, 'sharing_rate': 0}
#         cell_info = {**default_cell_info, **cell_info}
#         self.input_size = cell_info['input_size']
#         self.output_size = cell_info['output_size']
#         self.sharing_rate = cell_info['sharing_rate']
#         self.num_mode = cell_info['num_mode']
#         self.shared_size = round(self.sharing_rate * self.output_size)
#         self.free_size = self.output_size - self.shared_size
#         self.restricted_output_size = self.shared_size + self.free_size * self.num_mode
#         super(RLinearCell, self).__init__(self.input_size, self.restricted_output_size,
#                                           bias=cell_info['bias'])
#         self.register_buffer('shared_mask', torch.ones(self.shared_size))
#         # self.normalization = Normalization(cell_info['normalization'], self.restricted_output_size)
#         # self.activation = Activation(cell_info['activation'])
#
#     def forward(self, input):
#         mask = self.shared_mask.view(1, self.shared_mask.size(0)).expand(input.size(0), self.shared_mask.size(0))
#         mask = torch.cat((mask, config.PARAM['attr'].repeat_interleave(self.free_size, dim=1).detach()), dim=1).bool()
#         weight_mask = mask.view(mask.size(0), mask.size(1), 1)
#         weight = torch.masked_select(self.weight, weight_mask).view(input.size(0), self.output_size, self.input_size)
#         output = (input.view(input.size(0), 1, input.size(1)) * weight).sum(dim=2)
#         if self.bias is not None:
#             bias_mask = mask
#             bias = torch.masked_select(self.bias, bias_mask).view(input.size(0), self.output_size)
#             output = output + bias
#         return output


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


# class RBatchNorm1d(nn.BatchNorm1d):
#     def __init__(self, cell_info):
#         default_cell_info = {'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True,
#                              'sharing_rate': 1}
#         cell_info = {**default_cell_info, **cell_info}
#         self.input_size = cell_info['input_size']
#         self.eps = cell_info['eps']
#         self.momentum = cell_info['momentum']
#         self.affine = cell_info['affine']
#         self.track_running_stats = cell_info['track_running_stats']
#         self.sharing_rate = cell_info['sharing_rate']
#         self.num_mode = cell_info['num_mode']
#         self.shared_size = round(self.sharing_rate * self.input_size)
#         self.free_size = self.input_size - self.shared_size
#         self.restricted_input_size = self.shared_size + self.free_size * self.num_mode
#         super(RBatchNorm1d, self).__init__(self.restricted_input_size, eps=self.eps, momentum=self.momentum,
#                                            affine=self.affine, track_running_stats=self.track_running_stats)
#         self.register_buffer('shared_mask', torch.ones(self.shared_size))
#
#     def forward(self, input):
#         self._check_input_dim(input)
#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum
#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:
#                     exponential_average_factor = self.momentum
#         if self.training:
#             size = input.size()
#             size_prods = size[0]
#             for i in range(len(size) - 2):
#                 size_prods *= size[i + 2]
#             if size_prods == 1:
#                 raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(
#                 size))
#         mask = self.shared_mask.view(1, self.shared_mask.size(0)).expand(input.size(0), self.shared_mask.size(0))
#         mask = torch.cat((mask, config.PARAM['attr'].repeat_interleave(self.free_size, dim=1).detach()), dim=1).bool()
#         if self.training or not self.track_running_stats:
#             x = input.new_zeros(input.size(0) * self.restricted_input_size)
#             x[mask.view(-1)] = input.view(-1)
#             x = x.view(input.size(0), self.restricted_input_size)
#             n = mask.sum(dim=0)
#             mean_i = x.sum(dim=0) / n
#             var_i = (x.pow(2).sum(dim=0) / n - mean_i ** 2)
#             var_i[n > 1] = var_i[n > 1] * (n[n > 1] / (n[n > 1] - 1))
#             mean_s = torch.masked_select(mean_i, mask).view(input.size(0), self.input_size)
#             var_s = torch.masked_select(var_i, mask).view(input.size(0), self.input_size)
#             weight_s = torch.masked_select(self.weight, mask).view(input.size(0), self.input_size)
#             bias_s = torch.masked_select(self.bias, mask).view(input.size(0), self.input_size)
#             output = (input - mean_s) / torch.sqrt(var_s + self.eps) * weight_s + bias_s
#             self.running_mean[n > 1] = (1.0 - exponential_average_factor) * self.running_mean[n > 1] + \
#                                        exponential_average_factor * mean_i[n > 1]
#             self.running_var[n > 1] = (1.0 - exponential_average_factor) * self.running_var[n > 1] + \
#                                       exponential_average_factor * var_i[n > 1]
#         else:
#             mean_s = torch.masked_select(self.running_mean, mask).view(input.size(0), self.input_size)
#             var_s = torch.masked_select(self.running_var, mask).view(input.size(0), self.input_size)
#             weight_s = torch.masked_select(self.weight, mask).view(input.size(0), self.input_size)
#             bias_s = torch.masked_select(self.bias, mask).view(input.size(0), self.input_size)
#             output = (input - mean_s) / torch.sqrt(var_s + self.eps) * weight_s + bias_s
#         return output
#
#
# if __name__ == "__main__":
#     batch_size = 10
#     num_mode = batch_size
#     input_size = 6
#     output_size = 5
#     sharing_rate = 0.5
#     eps = 1e-05
#     input = torch.randn(batch_size, input_size)
#     cell_info = {'input_size': input_size, 'sharing_rate': sharing_rate, 'num_mode': num_mode}
#     m = RBatchNorm1d(cell_info)
#     m.train(True)
#     label = torch.arange(num_mode)
#     label = label.view(label.size(0), 1)
#     onehot = label.new_zeros(label.size(0), num_mode).float()
#     onehot.scatter_(1, label, 1)
#     config.PARAM['attr'] = onehot
#     for i in range(10):
#         output = m(input)


# class RBatchNorm2d(nn.BatchNorm2d):
#     def __init__(self, cell_info):
#         default_cell_info = {'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True,
#                              'sharing_rate': 1}
#         cell_info = {**default_cell_info, **cell_info}
#         self.input_size = cell_info['input_size']
#         self.eps = cell_info['eps']
#         self.momentum = cell_info['momentum']
#         self.affine = cell_info['affine']
#         self.track_running_stats = cell_info['track_running_stats']
#         self.sharing_rate = cell_info['sharing_rate']
#         self.num_mode = cell_info['num_mode']
#         self.shared_size = round(self.sharing_rate * self.input_size)
#         self.free_size = self.input_size - self.shared_size
#         self.restricted_input_size = self.shared_size + self.free_size * self.num_mode
#         super(RBatchNorm2d, self).__init__(self.restricted_input_size, eps=self.eps, momentum=self.momentum,
#                                            affine=self.affine, track_running_stats=self.track_running_stats)
#         self.register_buffer('shared_mask', torch.ones(self.shared_size))
#
#     def forward(self, input):
#         self._check_input_dim(input)
#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum
#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:
#                     exponential_average_factor = self.momentum
#         if self.training:
#             size = input.size()
#             size_prods = size[0]
#             for i in range(len(size) - 2):
#                 size_prods *= size[i + 2]
#             if size_prods == 1:
#                 raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(
#                 size))
#         mask = self.shared_mask.view(1, self.shared_mask.size(0)).expand(input.size(0), self.shared_mask.size(0))
#         mask = torch.cat((mask, config.PARAM['attr'].repeat_interleave(self.free_size, dim=1).detach()), dim=1).bool()
#         if self.training or not self.track_running_stats:
#             x = input.new_zeros(input.size(0) * self.restricted_input_size * input.size(2) * input.size(3))
#             x[mask.view(*mask.size(), 1, 1).expand(
#                 mask.size(0), mask.size(1), input.size(2), input.size(3)).reshape(-1)] = input.view(-1)
#             x = x.view(input.size(0), self.restricted_input_size, input.size(2), input.size(3))
#             n = mask.sum(dim=0) * input.size(2) * input.size(3)
#             mean_i = x.sum(dim=(0, 2, 3)) / n
#             var_i = (x.pow(2).sum(dim=(0, 2, 3)) / n - mean_i ** 2)
#             var_i[n > 1] = var_i[n > 1] * (n[n > 1] / (n[n > 1] - 1))
#             mean_s = torch.masked_select(mean_i, mask).view(input.size(0), self.input_size, 1, 1)
#             var_s = torch.masked_select(var_i, mask).view(input.size(0), self.input_size, 1, 1)
#             weight_s = torch.masked_select(self.weight, mask).view(input.size(0), self.input_size, 1, 1)
#             bias_s = torch.masked_select(self.bias, mask).view(input.size(0), self.input_size, 1, 1)
#             output = (input - mean_s) / torch.sqrt(var_s + self.eps) * weight_s + bias_s
#             self.running_mean[n > 1] = (1.0 - exponential_average_factor) * self.running_mean[n > 1] + \
#                                        exponential_average_factor * mean_i[n > 1]
#             self.running_var[n > 1] = (1.0 - exponential_average_factor) * self.running_var[n > 1] + \
#                                       exponential_average_factor * var_i[n > 1]
#         else:
#             mean_s = torch.masked_select(self.running_mean, mask).view(input.size(0), self.input_size, 1, 1)
#             var_s = torch.masked_select(self.running_var, mask).view(input.size(0), self.input_size, 1, 1)
#             weight_s = torch.masked_select(self.weight, mask).view(input.size(0), self.input_size, 1, 1)
#             bias_s = torch.masked_select(self.bias, mask).view(input.size(0), self.input_size, 1, 1)
#             output = (input - mean_s) / torch.sqrt(var_s + self.eps) * weight_s + bias_s
#         return output


# if __name__ == "__main__":
#     batch_size = 10
#     num_mode = batch_size
#     input_size = 6
#     output_size = 5
#     sharing_rate = 0.5
#     eps = 1e-05
#     input = torch.randn(batch_size, input_size, 4, 4)
#     cell_info = {'input_size': input_size, 'sharing_rate': sharing_rate, 'num_mode': num_mode}
#     m = RBatchNorm2d(cell_info)
#     m.train(True)
#     label = torch.arange(num_mode)
#     label = label.view(label.size(0), 1)
#     onehot = label.new_zeros(label.size(0), num_mode).float()
#     onehot.scatter_(1, label, 1)
#     config.PARAM['attr'] = onehot
#     output = m(input)
#     print(output.size())


# if __name__ == "__main__":
#     batch_size = 10
#     num_mode = batch_size
#     input_size = 6
#     output_size = 5
#     shape = (4, 4)
#     kernel_size = 3
#     stride = 1
#     padding = 1
#     sharing_rate = 0.5
#     eps = 1e-05
#     input = torch.randn(batch_size, input_size, *shape)
#
#     label = torch.arange(num_mode)
#     label = label.view(label.size(0), 1)
#     onehot = label.new_zeros(label.size(0), num_mode).float()
#     onehot.scatter_(1, label, 1)
#     config.PARAM['attr'] = onehot
#     m = nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
#     weight= m.weight
#     bias = m.bias
#     output = m(input)
#     print(input.size())
#     print(weight.size())
#     x = F.unfold(input, kernel_size, padding=1, stride=1)
#     print(x.transpose(1, 2).size())
#     print(weight.view(weight.size(0), -1).t().size())
#     t = x.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()) + bias
#     x2 = (x.transpose(1, 2).unsqueeze(3) * weight.view(weight.size(0), -1).t()).sum(2) + bias
#     print(x2.size())
#     print((t-x2).abs().max())
#     exit()
#     x = t.transpose(1, 2)
#     x = F.fold(x, shape, 1)
#     print(x.size())
#     print((output-x).abs().max())

# class RConv2dCell(nn.Conv2d):
#     def __init__(self, cell_info):
#         default_cell_info = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
#                              'padding_mode': 'zeros', 'sharing_rate': 1}
#         cell_info = {**default_cell_info, **cell_info}
#         self.input_size = cell_info['input_size']
#         self.output_size = cell_info['output_size']
#         self.sharing_rate = cell_info['sharing_rate']
#         self.num_mode = cell_info['num_mode']
#         self.shared_size = round(self.sharing_rate * self.output_size)
#         self.free_size = self.output_size - self.shared_size
#         self.restricted_output_size = self.shared_size + self.free_size * self.num_mode
#
#         super(RConv2dCell, self).__init__(cell_info['input_size'],  self.restricted_output_size, cell_info[
#         'kernel_size'],
#                                           stride=cell_info['stride'], padding=cell_info['padding'],
#                                           dilation=cell_info['dilation'], groups=cell_info['groups'],
#                                           bias=cell_info['bias'], padding_mode=cell_info['padding_mode'])
#         self.register_buffer('shared_mask', torch.ones(self.shared_size))
#         if cell_info['normalization'] == 'rbn':
#             self.normalization = RBatchNorm2d(
#                 {'input_size': self.output_size, 'sharing_rate': self.sharing_rate, 'num_mode': self.num_mode})
#         else:
#             self.normalization = Normalization(cell_info['normalization'], self.output_size)
#         # self.activation = Activation(cell_info['activation'])
#
#     def forward(self, input):
#         mask = self.shared_mask.view(1, self.shared_mask.size(0)).expand(input.size(0), self.shared_mask.size(0))
#         mask = torch.cat((mask, config.PARAM['attr'].repeat_interleave(self.free_size, dim=1).detach()), dim=1).bool()
#         weight_mask = mask.view(mask.size(0), mask.size(1), 1, 1, 1)
#         weight = torch.masked_select(self.weight, weight_mask).view(input.size(0), self.output_size, self.input_size,
#                                                                     *self.kernel_size)
#         x = F.unfold(input, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
#         print(x.transpose(1, 2).unsqueeze(3).size())
#         print(weight.view(weight.size(0), 1, weight.size(1), -1)
#                   .transpose(2, 3).size())
#         output = (x.transpose(1, 2).unsqueeze(3) * weight.view(weight.size(0), 1, weight.size(1), -1)
#                   .transpose(2, 3)).sum(2).transpose(1, 2)
#         output_shape = (math.floor((input.size(2) + 2 * self.padding[0] -
#                                     self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1),
#                         math.floor((input.size(3) + 2 * self.padding[1] -
#                                     self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
#         output = F.fold(output, output_shape, 1)
#         if self.bias is not None:
#             bias_mask = mask
#             bias = torch.masked_select(self.bias, bias_mask).view(input.size(0), self.output_size, 1, 1)
#             output = output + bias
#         return self.normalization(output)
#
#
# class RBatchNorm2d(nn.BatchNorm2d):
#     def __init__(self, cell_info):
#         default_cell_info = {'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True,
#                              'sharing_rate': 1}
#         cell_info = {**default_cell_info, **cell_info}
#         self.input_size = cell_info['input_size']
#         self.eps = cell_info['eps']
#         self.momentum = cell_info['momentum']
#         self.affine = cell_info['affine']
#         self.track_running_stats = cell_info['track_running_stats']
#         self.sharing_rate = cell_info['sharing_rate']
#         self.num_mode = cell_info['num_mode']
#         self.shared_size = round(self.sharing_rate * self.input_size)
#         self.free_size = self.input_size - self.shared_size
#         self.restricted_input_size = self.shared_size + self.free_size * self.num_mode
#         super(RBatchNorm2d, self).__init__(self.restricted_input_size, eps=self.eps, momentum=self.momentum,
#                                            affine=self.affine, track_running_stats=self.track_running_stats)
#         self.register_buffer('shared_mask', torch.ones(self.shared_size))
#
#     def forward(self, input):
#         self._check_input_dim(input)
#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum
#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:
#                     exponential_average_factor = self.momentum
#         if self.training:
#             size = input.size()
#             size_prods = size[0]
#             for i in range(len(size) - 2):
#                 size_prods *= size[i + 2]
#             if size_prods == 1:
#                 raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(
#                 size))
#         mask = self.shared_mask.view(1, self.shared_mask.size(0)).expand(input.size(0), self.shared_mask.size(0))
#         mask = torch.cat((mask, config.PARAM['attr'].repeat_interleave(self.free_size, dim=1).detach()), dim=1).bool()
#         if self.training or not self.track_running_stats:
#             x = input.new_zeros(input.size(0) * self.restricted_input_size * input.size(2) * input.size(3))
#             x[mask.view(*mask.size(), 1, 1).expand(
#                 mask.size(0), mask.size(1), input.size(2), input.size(3)).reshape(-1)] = input.view(-1)
#             x = x.view(input.size(0), self.restricted_input_size, input.size(2), input.size(3))
#             n = mask.sum(dim=0) * input.size(2) * input.size(3)
#             mean_i = x.sum(dim=(0, 2, 3)) / n
#             var_i = (x.pow(2).sum(dim=(0, 2, 3)) / n - mean_i ** 2)
#             var_i[n > 1] = var_i[n > 1] * (n[n > 1] / (n[n > 1] - 1))
#             mean_s = torch.masked_select(mean_i, mask).view(input.size(0), self.input_size, 1, 1)
#             var_s = torch.masked_select(var_i, mask).view(input.size(0), self.input_size, 1, 1)
#             weight_s = torch.masked_select(self.weight, mask).view(input.size(0), self.input_size, 1, 1)
#             bias_s = torch.masked_select(self.bias, mask).view(input.size(0), self.input_size, 1, 1)
#             output = (input - mean_s) / torch.sqrt(var_s + self.eps) * weight_s + bias_s
#             self.running_mean[n > 1] = (1.0 - exponential_average_factor) * self.running_mean[n > 1] + \
#                                        exponential_average_factor * mean_i[n > 1]
#             self.running_var[n > 1] = (1.0 - exponential_average_factor) * self.running_var[n > 1] + \
#                                       exponential_average_factor * var_i[n > 1]
#         else:
#             mean_s = torch.masked_select(self.running_mean, mask).view(input.size(0), self.input_size, 1, 1)
#             var_s = torch.masked_select(self.running_var, mask).view(input.size(0), self.input_size, 1, 1)
#             weight_s = torch.masked_select(self.weight, mask).view(input.size(0), self.input_size, 1, 1)
#             bias_s = torch.masked_select(self.bias, mask).view(input.size(0), self.input_size, 1, 1)
#             output = (input - mean_s) / torch.sqrt(var_s + self.eps) * weight_s + bias_s
#         return output
#
# if __name__ == "__main__":
#     batch_size = 10
#     num_mode = batch_size
#     input_size = 6
#     output_size = 5
#     shape = (4, 4)
#     kernel_size = 3
#     stride = 1
#     padding = 1
#     sharing_rate = 1
#     input = torch.randn(batch_size, input_size, *shape)
#
#     label = torch.arange(num_mode)
#     label = label.view(label.size(0), 1)
#     onehot = label.new_zeros(label.size(0), num_mode).float()
#     onehot.scatter_(1, label, 1)
#     config.PARAM['attr'] = onehot
#
#     m1 = nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
#     output1 = m1(input)
#     cell_info = {'input_size': input_size, 'output_size': output_size, 'kernel_size': kernel_size, 'stride': stride,
#                  'padding': padding, 'sharing_rate': sharing_rate, 'num_mode':num_mode, 'normalization':'rbn'}
#     m2 = RConv2dCell(cell_info)
#     # m2.weight.data.copy_(m1.weight.data)
#     # m2.bias.data.copy_(m1.bias.data)
#     output2 = m2(input)
#     print(output2.size())
#     # print((output1-output2).abs().max())


# if __name__ == "__main__":
#     batch_size = 1
#     num_mode = batch_size
#     input_size = 4
#     output_size = 2
#     shape = (2, 2)
#     kernel_size = 2
#     stride = 1
#     padding = 0
#     sharing_rate = 0.5
#     eps = 1e-05
#     # input = torch.arange(batch_size * input_size * shape[0] * shape[1]).float().view(batch_size, input_size,
#     # *shape) + 1
#     input = torch.tensor([[1,2],[3,4]], dtype=torch.float32).view(1,1,2,2).repeat(batch_size, input_size, 1, 1)
#     label = torch.arange(num_mode)
#     label = label.view(label.size(0), 1)
#     onehot = label.new_zeros(label.size(0), num_mode).float()
#     onehot.scatter_(1, label, 1)
#     config.PARAM['attr'] = onehot
#     m1 = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=False)
#     m1.weight.data = torch.tensor([[1,2],[3,4]], dtype=torch.float32).view(1,1,2,2).repeat(input_size, output_size,
#     1, 1)
#     # bias = m1.bias
#     output1 = m1(input)
#     print(input)
#     print(m1.weight)
#     print(output1)
#     # input2 = torch.tensor([[1,0,2],[0,0,0],[3,0,4]], dtype=torch.float32).view(1,1,3,3)
#     # input2 = F.conv_transpose2d(input, input.new_zeros(batch_size, input_size, stride, stride), stride=stride,
#     groups=input.size(1))
#     input2 = F.conv_transpose2d(input,F.pad(torch.ones(input_size,1,1,1),(1,1,1,1)),stride=stride, padding=1,
#     groups=input_size)
#     m2 = nn.Conv2d(input_size, output_size, kernel_size, 1, kernel_size - 1, bias=False)
#     m2.weight.data = torch.tensor([[4,3],[2,1]], dtype=torch.float32).view(1,1,2,2).repeat(output_size, input_size,
#     1, 1)
#     # m2.bias.data.copy_(bias.data)
#     print(m2.weight)
#     output2 = m2(input2)/input_size
#     print(output2)
#     print((output1 - output2).abs().max())
#     exit()
# x = F.unfold(input, kernel_size, padding=1, stride=1)
# print(x.transpose(1, 2).size())
# print(weight.view(weight.size(0), -1).t().size())
# t = x.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()) + bias
# x2 = (x.transpose(1, 2).unsqueeze(3) * weight.view(weight.size(0), -1).t()).sum(2) + bias
# print(x2.size())
# print((t-x2).abs().max())
# exit()
# x = t.transpose(1, 2)
# x = F.fold(x, shape, 1)
# print(x.size())
# print((output-x).abs().max())


# class RConvTranspose2dCell(nn.ConvTranspose2d):
#     def __init__(self, cell_info):
#         default_cell_info = {'stride': 1, 'padding': 0, 'output_padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
#                              'padding_mode': 'zeros', 'sharing_rate': 1}
#         cell_info = {**default_cell_info, **cell_info}
#         self.input_size = cell_info['input_size']
#         self.output_size = cell_info['output_size']
#         self.sharing_rate = cell_info['sharing_rate']
#         self.num_mode = cell_info['num_mode']
#         self.shared_size = round(self.sharing_rate * self.output_size)
#         self.free_size = self.output_size - self.shared_size
#         self.restricted_output_size = self.shared_size + self.free_size * self.num_mode
#         super(RConvTranspose2dCell, self).__init__(cell_info['input_size'], cell_info['output_size'],
#                                                    cell_info['kernel_size'],
#                                                    stride=cell_info['stride'], padding=cell_info['padding'],
#                                                    output_padding=cell_info['output_padding'],
#                                                    dilation=cell_info['dilation'], groups=cell_info['groups'],
#                                                    bias=cell_info['bias'], padding_mode=cell_info['padding_mode'])
#         self.register_buffer('shared_mask', torch.ones(self.shared_size))
#         # if cell_info['normalization'] == 'rbn':
#         #     self.normalization = RBatchNorm2d(
#         #         {'input_size': self.output_size, 'sharing_rate': self.sharing_rate, 'num_mode': self.num_mode})
#         # else:
#         #     self.normalization = Normalization(cell_info['normalization'], self.output_size)
#         # self.normalization = RBatchNorm2d(
#         #         {'input_size': self.output_size, 'sharing_rate': self.sharing_rate, 'num_mode': self.num_mode})
#         # self.activation = Activation(cell_info['activation'])
#
#     def forward(self, input, output_size=None):
#         if self.padding_mode != 'zeros':
#             raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')
#         x = F.conv_transpose2d(input, F.pad(torch.ones(self.input_size, 1, 1, 1), (1, 1, 1, 1)),
#                                stride=stride, padding=1, groups=self.input_size)
#         print(input.size(), x.size())
#         mask = self.shared_mask.view(1, self.shared_mask.size(0)).expand(input.size(0), self.shared_mask.size(0))
#         mask = torch.cat((mask, config.PARAM['attr'].repeat_interleave(self.free_size, dim=1).detach()), dim=1).bool()
#         weight_mask = mask.view(mask.size(0), 1, mask.size(1), 1, 1)
#         weight = torch.masked_select(self.weight, weight_mask).view(input.size(0), self.input_size, self.output_size,
#                                                                     *self.kernel_size)
#         weight = torch.flip(weight.transpose(1, 2), [3,4])
#         x = F.unfold(x, self.kernel_size, dilation=1, padding=(self.kernel_size[0] - self.padding[0] - 1,
#         self.kernel_size[1] - self.padding[1] - 1),
#                      stride=1)
#         output = (x.transpose(1, 2).unsqueeze(3) * weight.reshape(weight.size(0), 1, weight.size(1), -1)
#                                                               .transpose(2, 3)).sum(2).transpose(1, 2)
#         output_shape = (
#             (input.size(2) - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] -
#             1) +
#             self.output_padding[0] + 1,
#             (input.size(3) - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size[1] -
#             1) +
#             self.output_padding[1] + 1)
#         output = F.fold(output, output_shape, 1)
#         if self.bias is not None:
#             bias_mask = mask
#             bias = torch.masked_select(self.bias, bias_mask).view(input.size(0), self.output_size, 1, 1)
#             output = output + bias
#         # return self.normalization(output)
#         return output
#
# output1 = 0
# if __name__ == "__main__":
#     batch_size = 10
#     num_mode = batch_size
#     input_size = 1
#     output_size = 1
#     shape = (2, 2)
#     kernel_size = 4
#     stride = 2
#     padding = 1
#     sharing_rate = 1
#     eps = 1e-05
#     input = torch.randn(batch_size, input_size, *shape)
#     # input = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).view(1, 1, 2, 2).repeat(batch_size, input_size,
#     1, 1)
#     # input = torch.tensor([[[[1, 2], [3, 4]], [[1, 0], [0, 0]]]], dtype=torch.float32)
#     label = torch.arange(num_mode)
#     label = label.view(label.size(0), 1)
#     onehot = label.new_zeros(label.size(0), num_mode).float()
#     onehot.scatter_(1, label, 1)
#     config.PARAM['attr'] = onehot
#     m1 = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
#     # m1.weight.data = torch.tensor([[[[1, 2], [3, 4]], [[0, 0], [0, 0]]]], dtype=torch.float32).transpose(0,1)
#     output1 = m1(input)
#     cell_info = {'input_size': input_size, 'output_size': output_size, 'kernel_size': kernel_size, 'stride': stride,
#                  'padding': padding, 'sharing_rate': sharing_rate, 'num_mode': num_mode}
#     m2 = RConvTranspose2dCell(cell_info)
#     m2.weight.data.copy_(m1.weight.data)
#     m2.bias.data.copy_(m1.bias.data)
#     # print(input)
#     # print(m1.weight)
#     # print(m2.weight)
#     output2 = m2(input)
#     # print(output1)
#     # print(output2)
#     print(output2.size())
#     print((output1 - output2).abs().max())
#
# class RBatchNorm1d(nn.BatchNorm1d):
#     def __init__(self, cell_info):
#         default_cell_info = {'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True,
#                              'sharing_rate': 1, 'num_mode': 1}
#         cell_info = {**default_cell_info, **cell_info}
#         self.input_size = cell_info['input_size']
#         self.eps = cell_info['eps']
#         self.momentum = cell_info['momentum']
#         self.affine = cell_info['affine']
#         self.track_running_stats = cell_info['track_running_stats']
#         self.sharing_rate = cell_info['sharing_rate']
#         self.num_mode = cell_info['num_mode']
#         self.shared_size = round(self.sharing_rate * self.input_size)
#         self.free_size = self.input_size - self.shared_size
#         self.restricted_input_size = self.shared_size + self.free_size * self.num_mode
#         super(RBatchNorm1d, self).__init__(self.restricted_input_size, eps=self.eps, momentum=self.momentum,
#                                            affine=self.affine, track_running_stats=self.track_running_stats)
#         self.register_buffer('shared_mask', torch.ones(self.shared_size))
#
#     def forward(self, input):
#         self._check_input_dim(input)
#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum
#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:
#                     exponential_average_factor = self.momentum
#         if self.training:
#             size = input.size()
#             size_prods = size[0]
#             for i in range(len(size) - 2):
#                 size_prods *= size[i + 2]
#             if size_prods == 1:
#                 raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
#         mask = self.shared_mask.view(1, self.shared_mask.size(0)).expand(input.size(0), self.shared_mask.size(0))
#         mask = torch.cat((mask, config.PARAM['attr'].repeat_interleave(self.free_size, dim=1).detach()), dim=1).bool()
#         if self.training or not self.track_running_stats:
#             x = input.new_zeros(input.size(0) * self.restricted_input_size)
#             x[mask.view(-1)] = input.view(-1)
#             x = x.view(input.size(0), self.restricted_input_size)
#             n = mask.sum(dim=0)
#             mean_i = input.new_zeros(self.restricted_input_size)
#             var_i = input.new_zeros(self.restricted_input_size) + 1
#             mean_i[n > 1] = x.sum(dim=0)[n > 1] / n[n > 1]
#             var_i[n > 1] = (x - mean_i).pow(2).sum(dim=0)[n > 1] / (n[n > 1] - 1)
#             mean_s = torch.masked_select(mean_i, mask).view(input.size(0), self.input_size)
#             var_s = torch.masked_select(var_i, mask).view(input.size(0), self.input_size)
#             weight_s = torch.masked_select(self.weight, mask).view(input.size(0), self.input_size)
#             bias_s = torch.masked_select(self.bias, mask).view(input.size(0), self.input_size)
#             output = (input - mean_s) / torch.sqrt(var_s + self.eps) * weight_s + bias_s
#             self.running_mean[n > 1] = (1.0 - exponential_average_factor) * self.running_mean[n > 1] + \
#                                        exponential_average_factor * mean_i[n > 1]
#             self.running_var[n > 1] = (1.0 - exponential_average_factor) * self.running_var[n > 1] + \
#                                       exponential_average_factor * var_i[n > 1]
#         else:
#             mean_s = torch.masked_select(self.running_mean, mask).view(input.size(0), self.input_size)
#             var_s = torch.masked_select(self.running_var, mask).view(input.size(0), self.input_size)
#             weight_s = torch.masked_select(self.weight, mask).view(input.size(0), self.input_size)
#             bias_s = torch.masked_select(self.bias, mask).view(input.size(0), self.input_size)
#             output = (input - mean_s) / torch.sqrt(var_s + self.eps) * weight_s + bias_s
#         return output
#
#
#
# def idx2onehot(idx):
#     if config.PARAM['subset'] == 'label' or config.PARAM['subset'] == 'identity':
#         idx = idx.view(idx.size(0), 1)
#         onehot = idx.new_zeros(idx.size(0), config.PARAM['classes_size']).float()
#         onehot.scatter_(1, idx, 1)
#     else:
#         onehot = idx.float()
#     return onehot
#
# import matplotlib.pyplot as plt
# if __name__ == "__main__":
#     data_name = 'MNIST'
#     subset = 'label'
#     dataset = fetch_dataset(data_name, subset)
#     data_loader = make_data_loader(dataset)
#     input_size = 1024
#     sharing_rate = 1
#     num_mode = 10
#     config.PARAM['classes_size'] = 10
#     cell_info = {'input_size': input_size, 'sharing_rate': sharing_rate, 'num_mode': num_mode}
#     m1 = RBatchNorm1d(cell_info)
#     m2 = nn.BatchNorm1d(input_size)
#     m1.train(True)
#     m2.train(True)
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         input['img'] = input['img'].view(input['img'].size(0), -1)
#         input['img'].requires_grad = True
#         config.PARAM['attr'] = idx2onehot(input[subset])
#         input['img'].grad = None
#         output1 = m1(input['img'])
#         output1.mean().backward()
#         grad1 = input['img'].grad
#         input['img'].grad = None
#         # print(grad1.sum())
#         output2 = m2(input['img'])
#         output2.mean().backward()
#         grad2 = input['img'].grad
#         # print(grad2.sum())
#         print((grad1-grad2).abs().sum())
#         if i == 10:
#             break
#     # print((m1.running_mean - m2.running_mean).abs().max())
#     # print((m1.running_var - m2.running_var).abs().max())
#     # plt.plot((m1.running_mean - m2.running_mean).abs().numpy())
#     # plt.show()
#     m1.train(False)
#     m2.train(False)
#     for i, input in enumerate(data_loader['test']):
#         input = collate(input)
#         input['img'] = input['img'].view(input['img'].size(0), -1)
#         config.PARAM['attr'] = idx2onehot(input[subset])
#         output1 = m1(input['img'])
#         output2 = m2(input['img'])
#         # print((output1 - output2).abs().max())
#         # if i == 10:
#         #     break

# class Restriction(nn.Module):
#     def __init__(self, cell_info):
#         super(Restriction, self).__init__()
#         default_cell_info = {'sharing_rate': 1, 'num_mode': 1}
#         cell_info = {**default_cell_info, **cell_info}
#         self.input_size = cell_info['input_size']
#         self.sharing_rate = cell_info['sharing_rate']
#         self.num_mode = cell_info['num_mode']
#         self.mode_size = math.ceil(self.input_size * (1 - self.sharing_rate) / self.num_mode)
#         self.free_size = self.mode_size * self.num_mode
#         self.shared_size = self.input_size - self.free_size
#         embedding = torch.zeros(self.num_mode, self.input_size)
#         if self.shared_size > 0:
#             embedding[:, :self.shared_size] = 1
#         if self.free_size > 0:
#             idx = torch.arange(self.num_mode).repeat_interleave(self.mode_size, dim=0).view(1, -1)
#             embedding[:, self.shared_size:].scatter_(0, idx, 1)
#         self.register_buffer('embedding', embedding)
#
#     def forward(self, input):
#         embedding = config.PARAM['attr'].matmul(self.embedding)
#         embedding = embedding.view(*embedding.size(), *([1]*(input.dim()-2)))
#         output = input * embedding
#         return output
#
# if __name__ == "__main__":
#     input_size = 20
#     sharing_rate = 0.5
#     num_mode = 4
#     input = torch.randn(num_mode, input_size)
#     label = torch.arange(num_mode)
#     label = label.view(label.size(0), 1)
#     onehot = label.new_zeros(label.size(0), num_mode).float()
#     onehot.scatter_(1, label, 1)
#     print(onehot)
#     config.PARAM['attr'] = onehot
#     cell_info = {'cell':'Restriction', 'input_size':input_size, 'sharing_rate':sharing_rate, 'num_mode': num_mode}
#     m = Restriction(cell_info)
#     output = m(input)
#     print(output)

import itertools
import numpy as np

# def construct_codebook(N, M, K):
#     codebook = set()
#     sum = np.zeros(N, dtype=np.int64)
#     for i in range(K):
#         indices = np.arange(N)[::-1] if i == 0 else np.argsort(sum)
#         candidates = itertools.combinations(indices.tolist(), M)
#         for c in candidates:
#             prev_length = len(codebook)
#             code_c = np.zeros(N, dtype=np.int64)
#             code_c[list(c)] = 1
#             str_code = ''.join(str(cc) for cc in code_c.tolist())
#             codebook.add(str_code)
#             if len(codebook) > prev_length:
#                 sum += code_c
#                 break
#     codebook = sorted(list(codebook))
#     for i in range(len(codebook)):
#         codebook[i] = [int(c) for c in codebook[i]]
#     codebook = np.array(codebook)
#     return codebook
#
# if __name__ == "__main__":
#     N = 10
#     M = 5
#     K = 10
#     codebook = construct_codebook(N, M, K)
#     print(codebook)


# if __name__ == "__main__":
#     N = 5
#     M = 3
#     indices = np.arange(N)
#     candidates = itertools.combinations(indices.tolist(), M)
#     for c in candidates:
#         print(c, sum(c))

# some primes for tesing
# primes = [2]
# x = 3
# while x < 100000:
#     if all(x % p for p in primes):
#         primes.append(x)
#     x += 2

# def findc(seq, csum, clen):
#     def _findc(csum, clen, comb, idx):
#         if clen <= 0:
#             if csum == 0:
#                 yield comb
#             return
#         while idx < len(seq):
#             candidate = seq[idx]
#             if candidate > csum:
#                 return
#             for f in _findc(csum - candidate, clen - 1, comb + [candidate], idx + 1):
#                 yield f
#             idx += 1
#     return _findc(csum, clen, [], 0)


import numpy as np

def findc(seq, ind, csum, clen):
    def _findc(csum, clen, comb, index, idx):
        if clen <= 0:
            if csum == 0:
                yield comb, index
            return
        while idx < len(seq):
            comb_c = seq[idx]
            index_c = ind[idx]
            if comb_c > csum:
                return
            for g in _findc(csum - comb_c, clen - 1, comb + [comb_c], index + [index_c], idx + 1):
                yield g
            idx += 1
    return _findc(csum, clen, [], [], 0)


def make_codebook(N, M, K):
    codebook = set()
    sum = np.zeros(N, dtype=np.int64)
    for i in range(K):
        sorted_seq = sum[::-1] if i == 0 else np.sort(sum)
        sorted_ind = np.arange(N)[::-1] if i == 0 else np.argsort(sum)
        min_sum, max_sum = sorted_seq[:M].sum(), sorted_seq[-M:].sum()
        for s in range(min_sum, max_sum + 1):
            g = findc(sorted_seq, sorted_ind, s, M)
            prev_size = len(codebook)
            for (_, idx) in g:
                code_c = np.zeros(N, dtype=np.int64)
                code_c[idx] = 1
                str_code = ''.join(str(cc) for cc in code_c.tolist())
                codebook.add(str_code)
                if len(codebook) > prev_size:
                    sum += code_c
                    break
            if len(codebook) > prev_size:
                break
    codebook = sorted(list(codebook))
    for i in range(len(codebook)):
        codebook[i] = [int(c) for c in codebook[i]]
    codebook = np.array(codebook)
    return codebook


# if __name__ == "__main__":
#     M = 3
#     seq = np.array([3,2,1,1,3,4,1])
#     sorted_seq = np.sort(seq)
#     sorted_ind = np.argsort(seq)
#     print(seq)
#     print(sorted_seq)
#     print(sorted_ind)
#     min_sum, max_sum = sorted_seq[:M].sum(), sorted_seq[-M:].sum()
#     for s in range(min_sum, max_sum + 1):
#         for f in findc(sorted_seq, sorted_ind, s, M):
#             print(s, f)


if __name__ == "__main__":
    N = 256
    M = 128
    K = 1000
    codebook = make_codebook(N, M, K)
    print(codebook)

