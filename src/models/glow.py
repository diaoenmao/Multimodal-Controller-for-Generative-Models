import config
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.linalg as la
from .utils import make_model


class ActNorm2d(nn.Module):
    def __init__(self, num_features):
        super(ActNorm2d, self).__init__()
        self.num_features = num_features
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logscale = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = 1e-6
        self.initialized = False

    def initialize(self, x):
        if not self.training:
            return
        with torch.no_grad():
            loc = -torch.mean(x, dim=[0, 2, 3], keepdim=True)
            var = torch.mean((x + loc) ** 2, dim=[0, 2, 3], keepdim=True)
            logscale = (1 / (var.sqrt() + self.eps)).log()
            self.loc.data.copy_(loc.data)
            self.logscale.data.copy_(logscale.data)
            self.initialized = True

    def _center(self, input):
        if not config.PARAM['reverse']:
            return input + self.loc
        else:
            return input - self.loc

    def _scale(self, input, logdet=None):
        height, width = input.size(2), input.size(3)
        if not config.PARAM['reverse']:
            input = input * torch.exp(self.logscale)
        else:
            input = input * torch.exp(-self.logscale)
        if logdet is not None:
            dlogdet = height * width * torch.sum(self.logscale)
            if config.PARAM['reverse']:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return input, logdet

    def forward(self, input, logdet=None):
        if not self.initialized:
            self.initialize(input)
        if not config.PARAM['reverse']:
            input = self._center(input)
            input, logdet = self._scale(input, logdet)
        else:
            input, logdet = self._scale(input, logdet)
            input = self._center(input)
        return input, logdet


class InvertibleConv2d1x1(nn.Module):
    def __init__(self, num_channels):
        super(InvertibleConv2d1x1, self).__init__()
        self.num_channels = num_channels
        weight_init = np.linalg.qr(np.random.randn(num_channels, num_channels))[0].astype(np.float32)
        p, l, u = la.lu(weight_init)
        s = np.diag(u)
        sign_s = np.sign(s)
        logabs_s = np.log(np.abs(s))
        triu_u = np.triu(u, k=1)
        u_mask = np.triu(np.ones((num_channels, num_channels), dtype=np.float32), 1)
        l_mask = np.tril(np.ones((num_channels, num_channels), dtype=np.float32), -1)
        eye = np.eye(num_channels, num_channels, dtype=np.float32)
        self.register_buffer('p', torch.tensor(p))
        self.register_buffer('sign_s', torch.tensor(sign_s))
        self.register_buffer('u_mask', torch.tensor(u_mask))
        self.register_buffer('l_mask', torch.tensor(l_mask))
        self.register_buffer('eye', torch.tensor(eye))
        self.l = nn.Parameter(torch.tensor(l))
        self.logabs_s = nn.Parameter(torch.tensor(logabs_s))
        self.triu_u = nn.Parameter(torch.tensor(triu_u))

    def make_weight(self, input):
        height, width = input.size(2), input.size(3)
        l = self.l * self.l_mask + self.eye
        u = self.triu_u * self.u_mask + torch.diag(self.sign_s * torch.exp(self.logabs_s))
        dlogdet = height * width * torch.sum(self.logabs_s)
        if not config.PARAM['reverse']:
            weight = self.p @ l @ u
        else:
            weight = (self.p @ l @ u).inverse()
        weight = weight.unsqueeze(2).unsqueeze(3)
        return weight, dlogdet

    def forward(self, input, logdet=None):
        weight, dlogdet = self.make_weight(input)
        if not config.PARAM['reverse']:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
        return z, logdet


class LinearZeros(nn.Linear):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super(LinearZeros, self).__init__(in_channels, out_channels)
        self.logscale_factor = logscale_factor
        self.register_parameter("logscale", nn.Parameter(torch.zeros(out_channels)))
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logscale * self.logscale_factor)


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, do_actnorm=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=(not do_actnorm))
        self.weight.data.normal_(mean=0.0, std=0.05)
        self.do_actnorm = do_actnorm
        if not self.do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, logscale_factor=3):
        super(Conv2dZeros, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.logscale_factor = logscale_factor
        self.register_parameter("logscale", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logscale * self.logscale_factor)


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super(Permute2d, self).__init__()
        self.num_channels = num_channels
        self.indices = np.arange(self.num_channels - 1, -1, -1).astype(np.long)
        self.indices_inverse = np.zeros((self.num_channels), dtype=np.long)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i
        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        np.random.shuffle(self.indices)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input):
        if not config.PARAM['reverse']:
            return input[:, self.indices, :, :]
        else:
            return input[:, self.indices_inverse, :, :]


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logscale, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logscale ** 2
        """
        return -0.5 * (logscale * 2. + ((x - mean) ** 2) / torch.exp(logscale * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logscale, x):
        likelihood = GaussianDiag.likelihood(mean, logscale, x)
        return torch.sum(likelihood, dim=[1, 2, 3])

    @staticmethod
    def sample(mean, logscale, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logscale) * eps_std)
        return mean + torch.exp(logscale) * eps


def split_feature(tensor, type="split"):
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def cat_feature(tensor_a, tensor_b):
    return torch.cat((tensor_a, tensor_b), dim=1)


class Split2d(nn.Module):
    def __init__(self, num_channels, do_mc=False, num_mode=None, controller_rate=None):
        super(Split2d, self).__init__()
        self.conv = Conv2dZeros(num_channels // 2, num_channels)
        self.do_mc = do_mc
        if self.do_mc:
            self.mc = make_model({'cell': 'MultimodalController', 'input_size': num_channels, 'num_mode': num_mode,
                                  'controller_rate': controller_rate})

    def split2d_prior(self, z):
        if not config.PARAM['reverse']:
            h = self.conv(z)
            if self.do_mc:
                h = self.mc(h)
        else:
            if self.do_mc:
                z = self.mc(z)
            h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, input, logdet=0., eps_std=None):
        if not config.PARAM['reverse']:
            z1, z2 = split_feature(input, "split")
            mean, logscale = self.split2d_prior(z1)
            logdet = GaussianDiag.logp(mean, logscale, z2) + logdet
            return z1, logdet
        else:
            z1 = input
            mean, logscale = self.split2d_prior(z1)
            z2 = GaussianDiag.sample(mean, logscale, eps_std)
            z = cat_feature(z1, z2)
            return z, logdet


def squeeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    B, C, H, W = input.size(0), input.size(1), input.size(2), input.size(3)
    assert H % factor == 0 and W % factor == 0, "{}".format((H, W))
    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x


def unsqueeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    factor2 = factor ** 2
    if factor == 1:
        return input
    B, C, H, W = input.size(0), input.size(1), input.size(2), input.size(3)
    assert C % (factor2) == 0, "{}".format(C)
    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)
    return x


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super(SqueezeLayer, self).__init__()
        self.factor = factor

    def forward(self, input, logdet=None):
        if not config.PARAM['reverse']:
            output = squeeze2d(input, self.factor)
            return output, logdet
        else:
            output = unsqueeze2d(input, self.factor)
            return output, logdet


def flow(in_channels, out_channels, hidden_channels, do_mc=False, num_mode=None, controller_rate=None):
    if do_mc:
        f = nn.Sequential(
            Conv2d(in_channels, hidden_channels),
            nn.ReLU(inplace=False),
            make_model({'cell': 'MultimodalController', 'input_size': hidden_channels, 'num_mode': num_mode,
                        'controller_rate': controller_rate}),
            Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=False),
            make_model({'cell': 'MultimodalController', 'input_size': hidden_channels, 'num_mode': num_mode,
                        'controller_rate': controller_rate}),
            Conv2dZeros(hidden_channels, out_channels),
        )
    else:
        f = nn.Sequential(
            Conv2d(in_channels, hidden_channels),
            nn.ReLU(inplace=False),
            Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=False),
            Conv2dZeros(hidden_channels, out_channels))
    return f


class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet: (obj.reverse(z), logdet),
        "shuffle": lambda obj, z, logdet: (obj.shuffle(z), logdet),
        "invconv": lambda obj, z, logdet: obj.invconv(z, logdet)
    }

    def __init__(self, in_channels, hidden_channels, flow_permutation="invconv", flow_coupling="additive", do_mc=False,
                 num_mode=None, controller_rate=None):
        assert flow_coupling in FlowStep.FlowCoupling, \
            "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)
        assert flow_permutation in FlowStep.FlowPermutation, \
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super(FlowStep, self).__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.actnorm = ActNorm2d(in_channels)
        self.do_mc = do_mc
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv2d1x1(in_channels)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
        if flow_coupling == "additive":
            self.flow = flow(in_channels // 2, in_channels // 2, hidden_channels, self.do_mc, num_mode, controller_rate)
        elif flow_coupling == "affine":
            self.flow = flow(in_channels // 2, in_channels, hidden_channels, self.do_mc, num_mode, controller_rate)
        else:
            raise ValueError('Not valid flow mode')

    def forward(self, input, logdet=None):
        if not config.PARAM['reverse']:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        z, logdet = self.actnorm(input, logdet=logdet)
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](self, z, logdet)
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.flow(z1)
        elif self.flow_coupling == "affine":
            h = self.flow(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        else:
            raise ValueError('Not valid flow coupling mode')
        z = cat_feature(z1, z2)
        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.flow(z1)
        elif self.flow_coupling == "affine":
            h = self.flow(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        else:
            raise ValueError('Not valid flow coupling mode')
        z = cat_feature(z1, z2)
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](self, z, logdet)
        z, logdet = self.actnorm(z, logdet=logdet)
        return z, logdet


class FlowNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L, flow_permutation="invconv", flow_coupling="additive",
                 do_mc=False, num_mode=None, controller_rate=None):
        """
                             K                                      K
        --> [Squeeze] -> [FlowStep] -> [Split] -> [Squeeze] -> [FlowStep]
               ^                           v
               |          (L - 1)          |
               + --------------------------+
        """
        super(FlowNet, self).__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        self.L = L
        self.do_mc = do_mc
        C, H, W = image_shape
        for i in range(L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])
            # 2. K FlowStep
            for i in range(K):
                self.layers.append(
                    FlowStep(in_channels=C, hidden_channels=hidden_channels, flow_permutation=flow_permutation,
                             flow_coupling=flow_coupling, do_mc=self.do_mc, num_mode=num_mode,
                             controller_rate=controller_rate))
                self.output_shapes.append([-1, C, H, W])
            # 3. Split2d
            if i < L - 1:
                self.layers.append(
                    Split2d(num_channels=C, do_mc=self.do_mc, num_mode=num_mode, controller_rate=controller_rate))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, input, logdet=0., eps_std=None):
        if not config.PARAM['reverse']:
            return self.encode(input, logdet)
        else:
            return self.decode(input, eps_std)

    def encode(self, z, logdet=0.0):
        config.PARAM['reverse'] = False
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet)
        return z, logdet

    def decode(self, z, eps_std=None):
        config.PARAM['reverse'] = True
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, eps_std=eps_std)
            else:
                z, logdet = layer(z, logdet=0)
        return z


class Glow(nn.Module):
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()

    def __init__(self, image_shape, hidden_channels, K, L, flow_permutation, flow_coupling, learn_top, y_classes,
                 conditional):
        super(Glow, self).__init__()
        self.image_shape = image_shape
        self.hidden_channels = hidden_channels
        self.K = K
        self.L = L
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.learn_top = learn_top
        self.y_classes = y_classes
        self.conditional = conditional
        self.weight_y = 0.5
        self.flow = FlowNet(image_shape=image_shape, hidden_channels=hidden_channels, K=K, L=L,
                            flow_permutation=flow_permutation, flow_coupling=flow_coupling)
        # for prior
        if self.learn_top:
            C = self.flow.output_shapes[-1][1]
            self.top = Conv2dZeros(C * 2, C * 2)
        if self.conditional:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(self.y_classes, 2 * C)
            self.project_class = LinearZeros(C, self.y_classes)
        # register prior hidden
        self.register_parameter("prior_h", nn.Parameter(
            torch.zeros([config.PARAM['batch_size']['train'] // config.PARAM['world_size'],
                         self.flow.output_shapes[-1][1] * 2,
                         self.flow.output_shapes[-1][2],
                         self.flow.output_shapes[-1][3]])))

    def prior(self, y_onehot=None):
        B, C = self.prior_h.size(0), self.prior_h.size(1)
        h = self.prior_h.detach().clone()
        assert torch.sum(h) == 0.0
        if self.learn_top:
            h = self.top(h)
        if self.conditional:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot).view(B, C, 1, 1)
            h += yp
        return split_feature(h, "split")

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        config.PARAM['reverse'] = input['reverse'] if 'reverse' in input else False
        if not config.PARAM['reverse']:
            x = input['img']
            y_onehot = F.one_hot(input['label'], config.PARAM['classes_size']).float()
            output['z'], output['nll'], output['logits'] = self.normal_flow(x, y_onehot)
            # loss
            loss_generative = Glow.loss_generative(output['nll'])
            loss_classes = 0
            if self.conditional:
                loss_classes = Glow.loss_class(output['logits'], input['label'])
            output['loss'] = loss_generative + self.weight_y * loss_classes
        else:
            z = input['z']
            eps_std = input['eps_std']
            y_onehot = F.one_hot(input['label'], config.PARAM['classes_size']).float()
            output['img'] = self.reverse_flow(z, y_onehot, eps_std)
        return output

    def normal_flow(self, x, y_onehot):
        config.PARAM['reverse'] = False
        pixels = x.size(2) * x.size(3)
        z = x + torch.normal(mean=torch.zeros_like(x),
                             std=torch.ones_like(x) * (1. / 256.))
        logdet = torch.zeros_like(x[:, 0, 0, 0])
        logdet += float(-np.log(256.) * pixels)
        # encode
        z, objective = self.flow(z, logdet=logdet)
        # prior
        mean, logs = self.prior(y_onehot)
        objective += GaussianDiag.logp(mean, logs, z)
        if self.conditional:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None
        # return
        nll = (-objective) / float(np.log(2.) * pixels)
        return z, nll, y_logits

    def reverse_flow(self, z, y_onehot, eps_std):
        config.PARAM['reverse'] = True
        with torch.no_grad():
            mean, logs = self.prior(y_onehot)
            if z is None:
                z = GaussianDiag.sample(mean, logs, eps_std)
            x = self.flow(z, eps_std=eps_std)
        return x

    def set_actnorm_init(self, initialized=True):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.initialized = initialized
        return

    def generate_z(self, img):
        self.eval()
        B = config.PARAM['batch_size']['train']
        x = img.unsqueeze(0).repeat(B, 1, 1, 1).cuda()
        z, _, _ = self(x)
        self.train()
        return z[0].detach().cpu().numpy()

    def generate_attr_deltaz(self, dataset):
        assert "y_onehot" in dataset[0]
        self.eval()
        with torch.no_grad():
            B = config.PARAM['batch_size']['train']
            N = len(dataset)
            attrs_pos_z = [[0, 0] for _ in range(self.y_classes)]
            attrs_neg_z = [[0, 0] for _ in range(self.y_classes)]
            for i in range(0, N, B):
                j = min([i + B, N])
                # generate z for data from i to j
                xs = [dataset[k]["x"] for k in range(i, j)]
                while len(xs) < B:
                    xs.append(dataset[0]["x"])
                xs = torch.stack(xs).cuda()
                zs, _, _ = self(xs)
                for k in range(i, j):
                    z = zs[k - i].detach().cpu().numpy()
                    # append to different attrs
                    y = dataset[k]["y_onehot"]
                    for ai in range(self.y_classes):
                        if y[ai] > 0:
                            attrs_pos_z[ai][0] += z
                            attrs_pos_z[ai][1] += 1
                        else:
                            attrs_neg_z[ai][0] += z
                            attrs_neg_z[ai][1] += 1
                # break
            deltaz = []
            for ai in range(self.y_classes):
                if attrs_pos_z[ai][1] == 0:
                    attrs_pos_z[ai][1] = 1
                if attrs_neg_z[ai][1] == 0:
                    attrs_neg_z[ai][1] = 1
                z_pos = attrs_pos_z[ai][0] / float(attrs_pos_z[ai][1])
                z_neg = attrs_neg_z[ai][0] / float(attrs_neg_z[ai][1])
                deltaz.append(z_pos - z_neg)
        self.train()
        return deltaz

    @staticmethod
    def loss_generative(nll):
        # Generative loss
        return torch.mean(nll)

    @staticmethod
    def loss_multi_classes(y_logits, y_onehot):
        if y_logits is None:
            return 0
        else:
            return Glow.BCE(y_logits, y_onehot.float())

    @staticmethod
    def loss_class(y_logits, y):
        if y_logits is None:
            return 0
        else:
            return Glow.CE(y_logits, y.long())


def cglow():
    image_shape = config.PARAM['img_shape']
    hidden_channels = config.PARAM['hidden_size']
    K = config.PARAM['K']
    L = config.PARAM['L']
    flow_permutation = config.PARAM['flow_permutation']
    flow_coupling = config.PARAM['flow_coupling']
    learn_top = config.PARAM['learn_top']
    y_classes = config.PARAM['classes_size']
    conditional = True
    model = Glow(image_shape, hidden_channels, K, L, flow_permutation, flow_coupling, learn_top, y_classes, conditional)
    return model


class MCGlow(nn.Module):
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()

    def __init__(self, image_shape, hidden_channels, K, L, flow_permutation, flow_coupling, learn_top, y_classes,
                 controller_rate):
        super(MCGlow, self).__init__()
        self.image_shape = image_shape
        self.hidden_channels = hidden_channels
        self.K = K
        self.L = L
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.learn_top = learn_top
        self.y_classes = y_classes
        self.controller_rate = controller_rate
        self.weight_y = 0.5
        self.flow = FlowNet(image_shape=image_shape, hidden_channels=hidden_channels, K=K, L=L,
                            flow_permutation=flow_permutation, flow_coupling=flow_coupling, do_mc=True,
                            num_mode=y_classes, controller_rate=controller_rate)
        # for prior
        if self.learn_top:
            C = self.flow.output_shapes[-1][1]
            self.top = Conv2dZeros(C * 2, C * 2)
        # register prior hidden
        self.register_parameter("prior_h", nn.Parameter(
            torch.zeros([config.PARAM['batch_size']['train'] // config.PARAM['world_size'],
                         self.flow.output_shapes[-1][1] * 2,
                         self.flow.output_shapes[-1][2],
                         self.flow.output_shapes[-1][3]])))

    def prior(self):
        h = self.prior_h.detach().clone()
        assert torch.sum(h) == 0.0
        if self.learn_top:
            h = self.top(h)
        return split_feature(h, "split")

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        config.PARAM['reverse'] = input['reverse'] if 'reverse' in input else False
        if not config.PARAM['reverse']:
            x = input['img']
            config.PARAM['indicator'] = F.one_hot(input['label'], config.PARAM['classes_size']).float()
            output['z'], output['nll'] = self.normal_flow(x)
            # loss
            loss_generative = Glow.loss_generative(output['nll'])
            output['loss'] = loss_generative
        else:
            z = input['z']
            eps_std = input['eps_std']
            config.PARAM['indicator'] = F.one_hot(input['label'], config.PARAM['classes_size']).float()
            output['img'] = self.reverse_flow(z, eps_std)
        return output

    def normal_flow(self, x):
        config.PARAM['reverse'] = False
        pixels = x.size(2) * x.size(3)
        z = x + torch.normal(mean=torch.zeros_like(x),
                             std=torch.ones_like(x) * (1. / 256.))
        logdet = torch.zeros_like(x[:, 0, 0, 0])
        logdet += float(-np.log(256.) * pixels)
        # encode
        z, objective = self.flow(z, logdet=logdet)
        # prior
        mean, logs = self.prior()
        objective += GaussianDiag.logp(mean, logs, z)
        # return
        nll = (-objective) / float(np.log(2.) * pixels)
        return z, nll

    def reverse_flow(self, z, eps_std):
        config.PARAM['reverse'] = True
        with torch.no_grad():
            mean, logs = self.prior()
            if z is None:
                z = GaussianDiag.sample(mean, logs, eps_std)
            x = self.flow(z, eps_std=eps_std)
        return x

    def set_actnorm_init(self, initialized=True):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.initialized = initialized
        return

    def generate_z(self, img):
        self.eval()
        B = config.PARAM['batch_size']['train']
        x = img.unsqueeze(0).repeat(B, 1, 1, 1).cuda()
        z, _, _ = self(x)
        self.train()
        return z[0].detach().cpu().numpy()

    def generate_attr_deltaz(self, dataset):
        assert "y_onehot" in dataset[0]
        self.eval()
        with torch.no_grad():
            B = config.PARAM['batch_size']['train']
            N = len(dataset)
            attrs_pos_z = [[0, 0] for _ in range(self.y_classes)]
            attrs_neg_z = [[0, 0] for _ in range(self.y_classes)]
            for i in range(0, N, B):
                j = min([i + B, N])
                # generate z for data from i to j
                xs = [dataset[k]["x"] for k in range(i, j)]
                while len(xs) < B:
                    xs.append(dataset[0]["x"])
                xs = torch.stack(xs).cuda()
                zs, _, _ = self(xs)
                for k in range(i, j):
                    z = zs[k - i].detach().cpu().numpy()
                    # append to different attrs
                    y = dataset[k]["y_onehot"]
                    for ai in range(self.y_classes):
                        if y[ai] > 0:
                            attrs_pos_z[ai][0] += z
                            attrs_pos_z[ai][1] += 1
                        else:
                            attrs_neg_z[ai][0] += z
                            attrs_neg_z[ai][1] += 1
                # break
            deltaz = []
            for ai in range(self.y_classes):
                if attrs_pos_z[ai][1] == 0:
                    attrs_pos_z[ai][1] = 1
                if attrs_neg_z[ai][1] == 0:
                    attrs_neg_z[ai][1] = 1
                z_pos = attrs_pos_z[ai][0] / float(attrs_pos_z[ai][1])
                z_neg = attrs_neg_z[ai][0] / float(attrs_neg_z[ai][1])
                deltaz.append(z_pos - z_neg)
        self.train()
        return deltaz

    @staticmethod
    def loss_generative(nll):
        # Generative loss
        return torch.mean(nll)

    @staticmethod
    def loss_multi_classes(y_logits, y_onehot):
        if y_logits is None:
            return 0
        else:
            return Glow.BCE(y_logits, y_onehot.float())

    @staticmethod
    def loss_class(y_logits, y):
        if y_logits is None:
            return 0
        else:
            return Glow.CE(y_logits, y.long())


def mcglow():
    image_shape = config.PARAM['img_shape']
    hidden_channels = config.PARAM['hidden_size']
    K = config.PARAM['K']
    L = config.PARAM['L']
    flow_permutation = config.PARAM['flow_permutation']
    flow_coupling = config.PARAM['flow_coupling']
    learn_top = config.PARAM['learn_top']
    y_classes = config.PARAM['classes_size']
    controller_rate = config.PARAM['controller_rate']
    model = MCGlow(image_shape, hidden_channels, K, L, flow_permutation, flow_coupling, learn_top, y_classes,
                   controller_rate)
    return model