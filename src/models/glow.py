import config
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from .utils import make_model


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(kernel_size), \
        "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def uniform_binning_correction(x, n_bits=8):
    """Replaces x^i with q^i(x) = U(x, x + 1.0 / 256.0).
    Args:
        x: 4-D Tensor of shape (NCHW)
        n_bits: optional.
    Returns:
        x: x ~ U(x, x + 1.0 / 256)
        objective: Equivalent to -q(x)*log(q(x)).
    """
    b, c, h, w = x.size()
    n_bins = 2 ** n_bits
    chw = c * h * w
    x += torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return x, objective


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def gaussian_p(mean, logs, x):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
            k = 1 (Independent)
            Var = logs ** 2
    """
    c = math.log(2 * math.pi)
    return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + c)


def gaussian_likelihood(mean, logs, x):
    p = gaussian_p(mean, logs, x)
    return torch.sum(p, dim=[1, 2, 3])


def gaussian_sample(mean, logs, temperature=1):
    # Sample from Gaussian with temperature
    z = torch.normal(mean, torch.exp(logs) * temperature)
    return z


def squeeze2d(input, factor):
    if factor == 1:
        return input

    B, C, H, W = input.size()

    assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)

    return x


def unsqueeze2d(input, factor):
    if factor == 1:
        return input

    factor2 = factor ** 2

    B, C, H, W = input.size()

    assert C % (factor2) == 0, "C module factor squared is not 0"

    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)

    return x


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.
    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not initialized")

        with torch.no_grad():
            bias = - torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3],
                              keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.initialized.fill_(1)

    def _center(self, input, reverse=False):
        if reverse:
            return input - self.bias
        else:
            return input + self.bias

    def _scale(self, input, logdet=None, reverse=False):

        if reverse:
            input = input * torch.exp(-self.logs)
        else:
            input = input * torch.exp(self.logs)

        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply by number of pixels
            """
            b, c, h, w = input.shape

            dlogdet = torch.sum(self.logs) * h * w

            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        self._check_input_dim(input)

        if self.initialized.item() == 0:
            self.initialize_parameters(input)

        if reverse:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        else:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)

        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))


class LinearZeros(nn.Module):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.logscale_factor = logscale_factor

        self.logs = nn.Parameter(torch.zeros(out_channels))

    def forward(self, input):
        output = self.linear(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding="same", do_actnorm=True, weight_std=0.05):
        super().__init__()
        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=(not do_actnorm))

        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)

        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = self.conv(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding="same", logscale_factor=3):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        self.indices = torch.arange(self.num_channels - 1, -1, -1,
                                    dtype=torch.long)
        self.indices_inverse = torch.zeros((self.num_channels),
                                           dtype=torch.long)

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        shuffle_idx = torch.randperm(self.indices.shape[0])
        self.indices = self.indices[shuffle_idx]

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4

        if not reverse:
            input = input[:, self.indices, :, :]
            return input
        else:
            return input[:, self.indices_inverse, :, :]


class Split2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = Conv2dZeros(num_channels // 2, num_channels)

    def split2d_prior(self, z):
        h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, input, logdet=0., reverse=False, temperature=None):
        if reverse:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = gaussian_sample(mean, logs, temperature)
            z = torch.cat((z1, z2), dim=1)
            return z, logdet
        else:
            z1, z2 = split_feature(input, "split")
            mean, logs = self.split2d_prior(z1)
            logdet = gaussian_likelihood(mean, logs, z2) + logdet
            return z1, logdet


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            output = unsqueeze2d(input, self.factor)
        else:
            output = squeeze2d(input, self.factor)

        return output, logdet


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer('p', p)
            self.register_buffer('sign_s', sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


def get_block(in_channels, out_channels, hidden_channels, do_mc=False, num_mode=None, controller_rate=None):
    if do_mc:
        block = nn.Sequential(
            Conv2d(in_channels, hidden_channels),
            nn.ReLU(inplace=False),
            make_model({'cell': 'MultimodalController', 'input_size': hidden_channels, 'num_mode': num_mode,
                        'controller_rate': controller_rate}),
            Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
            nn.ReLU(inplace=False),
            make_model({'cell': 'MultimodalController', 'input_size': hidden_channels, 'num_mode': num_mode,
                        'controller_rate': controller_rate}),
            Conv2dZeros(hidden_channels, out_channels))
    else:
        block = nn.Sequential(
            Conv2d(in_channels, hidden_channels),
            nn.ReLU(inplace=False),
            Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
            nn.ReLU(inplace=False),
            Conv2dZeros(hidden_channels, out_channels))
    return block


class FlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels, actnorm_scale,
                 flow_permutation, flow_coupling, LU_decomposed, do_mc=False, num_mode=None, controller_rate=None):
        super().__init__()
        self.flow_coupling = flow_coupling

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels,
                                             LU_decomposed=LU_decomposed)
            self.flow_permutation = \
                lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = \
                lambda z, logdet, rev: (self.shuffle(z, rev), logdet)
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = \
                lambda z, logdet, rev: (self.reverse(z, rev), logdet)

        # 3. coupling
        if flow_coupling == "additive":
            self.block = get_block(in_channels // 2,
                                   in_channels // 2,
                                   hidden_channels, do_mc=do_mc, num_mode=num_mode, controller_rate=controller_rate)
        elif flow_coupling == "affine":
            self.block = get_block(in_channels // 2,
                                   in_channels,
                                   hidden_channels, do_mc=do_mc, num_mode=num_mode, controller_rate=controller_rate)

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L,
                 actnorm_scale, flow_permutation, flow_coupling,
                 LU_decomposed, do_mc=False, num_mode=None, controller_rate=None):
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K
        self.L = L

        C, H, W = image_shape

        for i in range(L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])

            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(in_channels=C, hidden_channels=hidden_channels, actnorm_scale=actnorm_scale,
                             flow_permutation=flow_permutation, flow_coupling=flow_coupling,
                             LU_decomposed=LU_decomposed, do_mc=do_mc, num_mode=num_mode,
                             controller_rate=controller_rate))
                self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if i < L - 1:
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, input, logdet=0., reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True,
                                  temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z


class Glow(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L, actnorm_scale,
                 flow_permutation, flow_coupling, LU_decomposed, do_mc=False, num_mode=None, controller_rate=None):
        super().__init__()
        self.flow = FlowNet(image_shape=image_shape, hidden_channels=hidden_channels, K=K, L=L,
                            actnorm_scale=actnorm_scale, flow_permutation=flow_permutation, flow_coupling=flow_coupling,
                            LU_decomposed=LU_decomposed, do_mc=do_mc, num_mode=num_mode,
                            controller_rate=controller_rate)
        self.num_mode = num_mode
        self.do_mc = do_mc
        C = self.flow.output_shapes[-1][1]
        self.learn_top_fn = Conv2dZeros(C * 2, C * 2)
        if not do_mc:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(num_mode, 2 * C)
        # self.register_buffer("prior_h",
        #                      torch.zeros([1,
        #                                   self.flow.output_shapes[-1][1] * 2,
        #                                   self.flow.output_shapes[-1][2],
        #                                   self.flow.output_shapes[-1][3]]))
        self.register_parameter("prior_h",
                                nn.Parameter(torch.zeros([1,
                                                          self.flow.output_shapes[-1][1] * 2,
                                                          self.flow.output_shapes[-1][2],
                                                          self.flow.output_shapes[-1][3]])))

    def preprocess(self, x, n_bits=8):
        n_bins = 2 ** n_bits
        x = (x + 1) / 2
        x = x * 255
        if n_bits < 8:
            x = torch.floor(x / 2 ** (8 - n_bits))
        x = x / n_bins - 0.5
        return x

    def postprocess(self, x, n_bits=8):
        n_bins = 2 ** n_bits
        x = torch.floor((x + 0.5) * n_bins)
        x = x * (256. / n_bins)
        x = torch.clamp(x, 0, 255)
        x = x / 255
        x = x * 2 - 1
        return x

    def prior(self, C):
        h = self.prior_h.repeat(C.size(0), 1, 1, 1)
        channels = h.size(1)
        h = self.learn_top_fn(h)
        if not self.do_mc:
            yp = self.project_ycond(config.PARAM['indicator'])
            h += yp.view(C.size(0), channels, 1, 1)
        return split_feature(h, "split")

    def forward(self, input, x=None, z=None):
        output = {}
        reverse = input['reverse'] if 'reverse' in input else False
        config.PARAM['indicator'] = F.one_hot(input['label'], config.PARAM['classes_size']).float()
        if reverse:
            z = input['z']
            temperature = input['temperature'] if 'temperature' in input else 1
            x = self.reverse_flow(z, config.PARAM['indicator'], temperature)
            output['img'] = self.postprocess(x)
        else:
            x = self.preprocess(input['img']).detach()
            output['z'], bpd = self.normal_flow(x, config.PARAM['indicator'])
            output['loss'] = bpd.mean()
        return output

    def normal_flow(self, x, onehot):
        b, c, h, w = x.shape
        x, logdet = uniform_binning_correction(x)
        z, objective = self.flow(x, logdet=logdet, reverse=False)
        mean, logs = self.prior(onehot)
        objective += gaussian_likelihood(mean, logs, z)
        bpd = (-objective) / (math.log(2.) * c * h * w)
        return z, bpd

    def reverse_flow(self, z, onehot, temperature):
        if z is None:
            mean, logs = self.prior(onehot)
            z = gaussian_sample(mean, logs, temperature)
        x = self.flow(z, temperature=temperature, reverse=True)
        return x

    def generate(self, C, x=None, temperature=1):
        config.PARAM['indicator'] = F.one_hot(C, config.PARAM['classes_size']).float()
        if x is None:
            mean, logs = self.prior(config.PARAM['indicator'])
            x = gaussian_sample(mean, logs, temperature)
        x = self.flow(x, temperature=temperature, reverse=True)
        generated = self.postprocess(x)
        return generated


def cglow():
    img_shape = config.PARAM['img_shape']
    hidden_size = config.PARAM['hidden_size']
    K = config.PARAM['K']
    L = config.PARAM['L']
    actnorm_scale = config.PARAM['actnorm_scale']
    flow_permutation = config.PARAM['flow_permutation']
    flow_coupling = config.PARAM['flow_coupling']
    LU_decomposed = config.PARAM['LU_decomposed']
    num_mode = config.PARAM['classes_size']
    model = Glow(img_shape, hidden_size, K, L, actnorm_scale, flow_permutation, flow_coupling, LU_decomposed,
                 do_mc=False, num_mode=num_mode, controller_rate=None)
    return model


def mcglow():
    img_shape = config.PARAM['img_shape']
    hidden_size = config.PARAM['hidden_size']
    K = config.PARAM['K']
    L = config.PARAM['L']
    actnorm_scale = config.PARAM['actnorm_scale']
    flow_permutation = config.PARAM['flow_permutation']
    flow_coupling = config.PARAM['flow_coupling']
    LU_decomposed = config.PARAM['LU_decomposed']
    num_mode = config.PARAM['classes_size']
    controller_rate = config.PARAM['controller_rate']
    model = Glow(img_shape, hidden_size, K, L, actnorm_scale, flow_permutation, flow_coupling, LU_decomposed,
                 do_mc=True, num_mode=num_mode, controller_rate=controller_rate)
    return model