import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.linalg as la
from math import log, pi, exp
from config import cfg
from modules import Wrapper, MultimodalController
from .utils import init_param


def logabs(x):
    return torch.log(torch.abs(x))


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class ActNorm(nn.Module):
    def __init__(self, input_size, logdet=True):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, input_size, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, input_size, 1, 1))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        if not self.training:
            return
        with torch.no_grad():
            mean = torch.mean(input, dim=[0, 2, 3], keepdim=True)
            std = torch.std(input, dim=[0, 2, 3], keepdim=True)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))
            self.initialized.fill_(1)
        return

    def forward(self, input):
        height, width = input.size(2), input.size(3)
        if self.initialized.item() == 0:
            self.initialize(input)
        log_abs = logabs(self.scale)
        logdet = height * width * torch.sum(log_abs)
        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        weight = torch.randn(input_size, input_size)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        height, width = input.size(2), input.size(3)
        out = F.conv2d(input, self.weight)
        logdet = height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        return out, logdet

    def reverse(self, output):
        return F.conv2d(output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class InvConv2dLU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        weight = np.random.randn(input_size, input_size)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T
        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)
        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        height, width = input.size(2), input.size(3)
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)
        return out, logdet

    def calc_weight(self):
        weight = (
                self.w_p
                @ (self.w_l * self.l_mask + self.l_eye)
                @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, output_size, 1, 1))

    def forward(self, input):
        out = self.conv(input)
        out = out * torch.exp(self.scale * 3)
        return out


class AffineCoupling(nn.Module):
    def __init__(self, input_size, hidden_size=512, affine=True, num_mode=None, controller_rate=None):
        super().__init__()
        self.affine = affine
        self.net = nn.Sequential(
            Wrapper(nn.Conv2d(input_size // 2, hidden_size, 3, padding=1)),
            Wrapper(ActNorm(hidden_size, logdet=False)),
            Wrapper(nn.ReLU(inplace=True)),
            MultimodalController(hidden_size, num_mode, controller_rate),
            Wrapper(nn.Conv2d(hidden_size, hidden_size, 1)),
            Wrapper(ActNorm(hidden_size, logdet=False)),
            Wrapper(nn.ReLU(inplace=True)),
            MultimodalController(hidden_size, num_mode, controller_rate),
            Wrapper(ZeroConv2d(hidden_size, input_size if self.affine else input_size // 2)),
        )
        self.net[0].module.weight.data.normal_(0, 0.05)
        self.net[0].module.bias.data.zero_()
        self.net[4].module.weight.data.normal_(0, 0.05)
        self.net[4].module.bias.data.zero_()

    def forward(self, input, indicator):
        in_a, in_b = input.chunk(2, 1)
        if self.affine:
            log_s, t = self.net((in_a, indicator))[0].chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
            logdet = torch.sum(torch.log(s).view(input.size(0), -1), 1)
        else:
            net_out = self.net((in_a, indicator))[0]
            out_b = in_b + net_out
            logdet = None
        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output, indicator):
        out_a, out_b = output.chunk(2, 1)
        if self.affine:
            log_s, t = self.net((out_a, indicator))[0].chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            net_out = self.net((out_a, indicator))[0]
            in_b = out_b - net_out
        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, input_size, hidden_size, affine=True, conv_lu=True, num_mode=None, controller_rate=None):
        super().__init__()
        self.actnorm = ActNorm(input_size)
        if conv_lu:
            self.invconv = InvConv2dLU(input_size)
        else:
            self.invconv = InvConv2d(input_size)
        self.coupling = AffineCoupling(input_size, hidden_size, affine, num_mode, controller_rate)

    def forward(self, input, indicator):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out, indicator)
        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, output, indicator):
        input = self.coupling.reverse(output, indicator)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input


class Block(nn.Module):
    def __init__(self, input_size, hidden_size, K, split=True, affine=True, conv_lu=True, num_mode=None,
                 controller_rate=None):
        super().__init__()
        squeeze_dim = input_size * 4
        self.flows = nn.ModuleList()
        for i in range(K):
            self.flows.append(Flow(squeeze_dim, hidden_size, affine, conv_lu, num_mode, controller_rate))
        self.split = split
        if split:
            self.prior = ZeroConv2d(input_size * 2, input_size * 4)
        else:
            self.prior = ZeroConv2d(input_size * 4, input_size * 8)

    def forward(self, input, indicator):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.reshape(b_size, n_channel * 4, height // 2, width // 2)
        logdet = 0
        for flow in self.flows:
            out, det = flow(out, indicator)
            logdet = logdet + det
        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
        else:
            zero = torch.zeros_like(out)
            h = self.prior(zero)
            mean, log_sd = h.chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out
        return out, logdet, log_p, z_new

    def reverse(self, output, indicator, eps=None, reconstruct=False):
        input = output
        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps
        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)
            else:
                zero = torch.zeros_like(input)
                h = self.prior(zero)
                mean, log_sd = h.chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z
        for flow in self.flows[::-1]:
            input = flow.reverse(input, indicator)
        b_size, n_channel, height, width = input.shape
        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.reshape(b_size, n_channel // 4, height * 2, width * 2)
        return unsqueezed


class MCGlow(nn.Module):
    def __init__(self, data_shape, hidden_size, K, L, affine=True, conv_lu=True, num_mode=None, controller_rate=0.5):
        super().__init__()
        self.data_shape = data_shape
        self.K = K
        self.L = L
        self.num_mode = num_mode
        self.controller_rate = controller_rate
        self.blocks = nn.ModuleList()
        input_size = self.data_shape[0]
        for i in range(L - 1):
            self.blocks.append(Block(input_size, hidden_size, K, True, affine, conv_lu, num_mode, controller_rate))
            input_size *= 2
        self.blocks.append(Block(input_size, hidden_size, K, False, affine, conv_lu, num_mode, controller_rate))

    def loss_fn(self, log_p, logdet):
        n_pixel = np.prod(self.data_shape)
        loss = -float(log(256.) * n_pixel)
        loss = loss + logdet + log_p
        loss = -loss / float(log(2.) * n_pixel)
        if self.training:
            loss[torch.isnan(loss)] = 0
        else:
            loss = loss[~torch.isnan(loss)]
        loss = loss.mean()
        return loss

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        indicator = F.one_hot(input['label'], cfg['classes_size']).float()
        x = input['img'] * 0.5
        x = x + torch.rand_like(x) / 256
        z = []
        log_p_sum = 0
        logdet = 0
        for block in self.blocks:
            x, det, log_p, z_new = block(x, indicator)
            z.append(z_new)
            logdet = logdet + det
            if log_p is not None:
                log_p_sum = log_p_sum + log_p
        nll = self.loss_fn(log_p_sum, logdet)
        output['loss'] = nll
        output['z'] = z
        return output

    def reverse(self, input):
        output = {}
        indicator = F.one_hot(input['label'], cfg['classes_size']).float()
        z = input['z']
        x = None
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                x = block.reverse(z[-1], indicator, eps=z[-1], reconstruct=input['reconstruct'])
            else:
                x = block.reverse(x, indicator, eps=z[-(i + 1)], reconstruct=input['reconstruct'])
        output['img'] = torch.clamp(x, -.5, .5) * 2
        return output

    def make_z_shapes(self):
        C, H, W = self.data_shape
        z_shapes = []
        for i in range(self.L - 1):
            H, W = H // 2, W // 2
            C *= 2
            z_shapes.append((C, H, W))
        H, W = H // 2, W // 2
        z_shapes.append((C * 4, H, W))
        return z_shapes

    def generate(self, C, x=None, temperature=1):
        input = {}
        if x is None:
            z_shapes = self.make_z_shapes()
            x = []
            for i in range(len(z_shapes)):
                x_i = torch.randn([C.size(0), *z_shapes[i]], device=cfg['device']) * temperature
                x.append(x_i)
        input['z'] = x
        input['reconstruct'] = False
        input['label'] = C
        generated = self.reverse(input)['img']
        return generated


def mcglow():
    data_shape = cfg['data_shape']
    hidden_size = cfg['glow']['hidden_size']
    K = cfg['glow']['K']
    L = cfg['glow']['L']
    affine = cfg['glow']['affine']
    conv_lu = cfg['glow']['conv_lu']
    num_mode = cfg['classes_size']
    controller_rate = cfg['controller_rate']
    model = MCGlow(data_shape, hidden_size, K, L, affine, conv_lu, num_mode, controller_rate)
    model.apply(init_param)
    return model