import config
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.linalg as la
from .utils import make_model
from math import log, pi, exp


def logabs(x):
    return torch.log(torch.abs(x))


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super(ActNorm, self).__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.initialized = False
        self.logdet = logdet

    def initialize(self, input):
        if not self.training:
            return
        with torch.no_grad():
            mean = torch.mean(input, dim=[0, 2, 3], keepdim=True)
            std = torch.std(input, dim=[0, 2, 3], keepdim=True)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))
            self.initialized = True

    def forward(self, input):
        height, width = input.size(2), input.size(3)
        if not self.initialized:
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
    def __init__(self, in_channel):
        super(InvConv2d, self).__init__()
        weight = torch.randn(in_channel, in_channel)
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
    def __init__(self, in_channel):
        super(InvConv2dLU, self).__init__()
        weight = np.random.randn(in_channel, in_channel)
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
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(ZeroConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = self.conv(input)
        out = out * torch.exp(self.scale * 3)
        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, hidden_size=512, affine=True, do_mc=False, num_mode=None,
                 controller_rate=None):
        super(AffineCoupling, self).__init__()
        self.affine = affine
        if do_mc:
            self.net = nn.Sequential(
                nn.Conv2d(in_channel // 2, hidden_size, 3, padding=1),
                ActNorm(hidden_size, logdet=False),
                nn.ReLU(inplace=True),
                make_model({'cell': 'MultimodalController', 'input_size': hidden_size, 'num_mode': num_mode,
                            'controller_rate': controller_rate}),
                nn.Conv2d(hidden_size, hidden_size, 1),
                ActNorm(hidden_size, logdet=False),
                nn.ReLU(inplace=True),
                make_model({'cell': 'MultimodalController', 'input_size': hidden_size, 'num_mode': num_mode,
                            'controller_rate': controller_rate}),
                ZeroConv2d(hidden_size, in_channel if self.affine else in_channel // 2),
            )
            self.net[0].weight.data.normal_(0, 0.05)
            self.net[0].bias.data.zero_()
            self.net[4].weight.data.normal_(0, 0.05)
            self.net[4].bias.data.zero_()
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channel // 2, hidden_size, 3, padding=1),
                ActNorm(hidden_size, logdet=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_size, hidden_size, 1),
                ActNorm(hidden_size, logdet=False),
                nn.ReLU(inplace=True),
                ZeroConv2d(hidden_size, in_channel if self.affine else in_channel // 2),
            )
            self.net[0].weight.data.normal_(0, 0.05)
            self.net[0].bias.data.zero_()
            self.net[3].weight.data.normal_(0, 0.05)
            self.net[3].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
            logdet = torch.sum(torch.log(s).view(input.size(0), -1), 1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None
        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out
        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, hidden_size, affine=True, conv_lu=True, do_mc=False, num_mode=None,
                 controller_rate=None):
        super(Flow, self).__init__()
        self.actnorm = ActNorm(in_channel)
        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)
        self.coupling = AffineCoupling(in_channel, hidden_size, affine=affine, do_mc=do_mc, num_mode=num_mode,
                                       controller_rate=controller_rate)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)
        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input


class Block(nn.Module):
    def __init__(self, in_channel, hidden_size, K, num_mode, split=True, affine=True, conv_lu=True, do_mc=False,
                 controller_rate=None):
        super(Block, self).__init__()
        squeeze_dim = in_channel * 4
        self.do_mc = do_mc
        self.flows = nn.ModuleList()
        for i in range(K):
            self.flows.append(Flow(squeeze_dim, hidden_size, affine=affine, conv_lu=conv_lu, do_mc=do_mc,
                                   num_mode=num_mode, controller_rate=controller_rate))
        self.split = split
        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)
        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)
        if not self.do_mc:
            self.embedding = ZeroConv2d(num_mode, in_channel * 8, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.reshape(b_size, n_channel * 4, height // 2, width // 2)
        logdet = 0
        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det
        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
        else:
            zero = torch.zeros_like(out)
            h = self.prior(zero)
            if not self.do_mc:
                h_indicator = self.embedding(config.PARAM['indicator'].view([*config.PARAM['indicator'].size(), 1, 1]))
                h += h_indicator
            mean, log_sd = h.chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out
        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
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
                if not self.do_mc:
                    h_indicator = self.embedding(
                        config.PARAM['indicator'].view([*config.PARAM['indicator'].size(), 1, 1]))
                    h += h_indicator
                mean, log_sd = h.chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z
        for flow in self.flows[::-1]:
            input = flow.reverse(input)
        b_size, n_channel, height, width = input.shape
        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.reshape(b_size, n_channel // 4, height * 2, width * 2)
        return unsqueezed


class CGlow(nn.Module):
    def __init__(self, img_shape, hidden_size, K, L, num_mode, affine=True, conv_lu=True):
        super(CGlow, self).__init__()
        self.img_shape = img_shape
        self.K = K
        self.L = L
        self.num_mode = num_mode
        self.blocks = nn.ModuleList()
        in_channel = self.img_shape[0]
        for i in range(L - 1):
            self.blocks.append(Block(in_channel, hidden_size, K, num_mode, split=True, affine=affine, conv_lu=conv_lu))
            in_channel *= 2
        self.blocks.append(Block(in_channel, hidden_size, K, num_mode, split=False, affine=affine, conv_lu=conv_lu))
        # self.classifier = ZeroConv2d(4 * in_channel, num_mode, kernel_size=1, stride=1, padding=0)

    def loss_fn(self, log_p, logdet):
        n_pixel = np.prod(self.img_shape)
        loss = -float(log(256.) * n_pixel)
        loss = loss + logdet + log_p
        loss = -loss / float(log(2.) * n_pixel)
        return loss.mean()

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        config.PARAM['indicator'] = F.one_hot(input['label'], config.PARAM['classes_size']).float()
        x = input['img']
        x = x + torch.rand_like(x) / 256
        z = []
        log_p_sum = 0
        logdet = 0
        for block in self.blocks:
            x, det, log_p, z_new = block(x)
            z.append(z_new)
            logdet = logdet + det
            if log_p is not None:
                log_p_sum = log_p_sum + log_p
        nll = self.loss_fn(log_p_sum, logdet)
        # output['logits'] = F.adaptive_avg_pool2d(self.classifier(z_new), 1).squeeze()
        # classification_loss = F.cross_entropy(output['logits'], input['label'])
        # output['loss'] = nll + config.PARAM['classification_loss_weight'] * classification_loss
        output['loss'] = nll
        output['z'] = z
        return output

    def reverse(self, input):
        output = {}
        config.PARAM['indicator'] = F.one_hot(input['label'], config.PARAM['classes_size']).float()
        z = input['z']
        x = None
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                x = block.reverse(z[-1], z[-1], reconstruct=input['reconstruct'])
            else:
                x = block.reverse(x, z[-(i + 1)], reconstruct=input['reconstruct'])
        output['img'] = torch.clamp(x, -1, 1)
        return output

    def make_z_shapes(self):
        C, H, W = config.PARAM['img_shape']
        z_shapes = []
        for i in range(self.L - 1):
            H, W = H // 2, W // 2
            C *= 2
            z_shapes.append((C, H, W))
        H, W = H // 2, W // 2
        z_shapes.append((C * 4, H, W))
        return z_shapes

    def generate(self, x, C):
        input = {}
        input['z'] = x
        input['reconstruct'] = False
        input['label'] = C
        generated = self.reverse(input)['img']
        return generated


def cglow():
    img_shape = config.PARAM['img_shape']
    hidden_size = config.PARAM['hidden_size']
    K = config.PARAM['K']
    L = config.PARAM['L']
    affine = config.PARAM['affine']
    conv_lu = config.PARAM['conv_lu']
    num_mode = config.PARAM['classes_size']
    model = CGlow(img_shape, hidden_size, K, L, num_mode, affine, conv_lu)
    return model


class MCGlow(nn.Module):
    def __init__(self, img_shape, hidden_size, K, L, num_mode, controller_rate, affine=True, conv_lu=True):
        super(MCGlow, self).__init__()
        self.img_shape = img_shape
        self.K = K
        self.L = L
        self.num_mode = num_mode
        self.controller_rate = controller_rate
        self.blocks = nn.ModuleList()
        in_channel = self.img_shape[0]
        for i in range(L - 1):
            self.blocks.append(
                Block(in_channel, hidden_size, K, num_mode, split=True, affine=affine, conv_lu=conv_lu, do_mc=True,
                      controller_rate=controller_rate))
            in_channel *= 2
        self.blocks.append(
            Block(in_channel, hidden_size, K, num_mode, split=False, affine=affine, conv_lu=conv_lu, do_mc=True,
                  controller_rate=controller_rate))
        # self.classifier = ZeroConv2d(4 * in_channel, num_mode, kernel_size=1, stride=1, padding=0)

    def loss_fn(self, log_p, logdet):
        n_pixel = np.prod(self.img_shape)
        loss = -float(log(256.) * n_pixel)
        loss = loss + logdet + log_p
        loss = -loss / float(log(2.) * n_pixel)
        return loss.mean()

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        config.PARAM['indicator'] = F.one_hot(input['label'], config.PARAM['classes_size']).float()
        x = input['img']
        x = x + torch.rand_like(x) / 256
        z = []
        log_p_sum = 0
        logdet = 0
        for block in self.blocks:
            x, det, log_p, z_new = block(x)
            z.append(z_new)
            logdet = logdet + det
            if log_p is not None:
                log_p_sum = log_p_sum + log_p
        nll = self.loss_fn(log_p_sum, logdet)
        # output['logits'] = F.adaptive_avg_pool2d(self.classifier(z_new), 1).squeeze()
        # classification_loss = F.cross_entropy(output['logits'], input['label'])
        # output['loss'] = nll + config.PARAM['classification_loss_weight'] * classification_loss
        output['loss'] = nll
        output['z'] = z
        return output

    def reverse(self, input):
        output = {}
        config.PARAM['indicator'] = F.one_hot(input['label'], config.PARAM['classes_size']).float()
        z = input['z']
        x = None
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                x = block.reverse(z[-1], z[-1], reconstruct=input['reconstruct'])
            else:
                x = block.reverse(x, z[-(i + 1)], reconstruct=input['reconstruct'])
        output['img'] = torch.clamp(x, -1, 1)
        return output

    def make_z_shapes(self):
        C, H, W = config.PARAM['img_shape']
        z_shapes = []
        for i in range(self.L - 1):
            H, W = H // 2, W // 2
            C *= 2
            z_shapes.append((C, H, W))
        H, W = H // 2, W // 2
        z_shapes.append((C * 4, H, W))
        return z_shapes

    def generate(self, x, C):
        input = {}
        input['z'] = x
        input['reconstruct'] = False
        input['label'] = C
        generated = self.reverse(input)['img']
        return generated


def mcglow():
    img_shape = config.PARAM['img_shape']
    hidden_size = config.PARAM['hidden_size']
    K = config.PARAM['K']
    L = config.PARAM['L']
    affine = config.PARAM['affine']
    conv_lu = config.PARAM['conv_lu']
    num_mode = config.PARAM['classes_size']
    controller_rate = config.PARAM['controller_rate']
    model = MCGlow(img_shape, hidden_size, K, L, num_mode, controller_rate, affine, conv_lu)
    return model