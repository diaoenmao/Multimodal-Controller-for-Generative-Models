import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_model


class GatedActivation(nn.Module):
    def __init__(self):
        super(GatedActivation, self).__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)


class ConditionalGatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, input_dim, output_dim, kernel, residual, n_classes):
        super(ConditionalGatedMaskedConv2d, self).__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual
        self.class_cond_embedding = nn.Embedding(n_classes, 2 * output_dim)
        kernel_shp = (kernel // 2 + 1, kernel)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(input_dim, 2 * output_dim, kernel_shp, 1, padding_shp)
        self.vert_to_horiz = nn.Conv2d(2 * output_dim, 2 * output_dim, 1)
        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(input_dim, 2 * output_dim, kernel_shp, 1, padding_shp)
        self.horiz_resid = nn.Conv2d(output_dim, output_dim, 1)
        self.gate = GatedActivation()
        self.bn_v = nn.BatchNorm2d(output_dim)
        self.bn_h = nn.BatchNorm2d(output_dim)

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()
        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert + h[:, :, None, None])
        out_v = self.bn_v(out_v)
        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)
        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.bn_h(self.horiz_resid(out)) + x_h
        else:
            out_h = self.bn_h(self.horiz_resid(out))
        return out_v, out_h


class ConditionalGatedPixelCNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=256, n_layers=15, n_classes=10):
        super(ConditionalGatedPixelCNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        # Initial block with Mask-A convolution
        self.layers.append(ConditionalGatedMaskedConv2d('A', input_dim, hidden_dim, 7, False, n_classes))
        # Rest with Mask-B convolutions
        for i in range(n_layers - 1):
            self.layers.append(ConditionalGatedMaskedConv2d('B', hidden_dim, hidden_dim, 3, True, n_classes))
        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim * output_dim, 1)
        )

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['img']
        label = input['label']
        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)
        x = self.output_conv(x_h)
        output['logits'] = x.view(x.size(0), self.output_dim, self.input_dim, x.size(2), x.size(3))
        output['loss'] = F.cross_entropy(output['logits'], ((input['img'] + 1) / 2 * 255).long().detach())
        return output

    def generate(self, x, C):
        input = {'img': x, 'label': C}
        for i in range(x.size(1)):
            for j in range(x.size(2)):
                output = self.forward(input)
                logits = output['logits'].permute(0, 2, 1, 3, 4)
                logits = logits.reshape(-1, logits.size(2), logits.size(3), logits.size(4))
                probs = F.softmax(logits[:, :, i, j], -1)
                samples = probs.multinomial(1).squeeze().view(input['img'].size(0), -1).float() / 255 * 2 - 1
                input['img'].data[:, :, i, j].copy_(samples)
        return input['img']


class MCGatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, input_dim, output_dim, kernel, residual, n_classes, controller_rate):
        super(MCGatedMaskedConv2d, self).__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual
        kernel_shp = (kernel // 2 + 1, kernel)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = make_model(
            {'cell': 'MCConv2dCell', 'input_size': input_dim, 'output_size': 2 * output_dim, 'kernel_size': kernel_shp,
             'stride': 1, 'padding': padding_shp, 'bias': True, 'normalization': 'none', 'activation': 'none',
             'num_mode': n_classes, 'controller_rate': controller_rate})
        self.vert_to_horiz = make_model(
            {'cell': 'MCConv2dCell', 'input_size': 2 * output_dim, 'output_size': 2 * output_dim, 'kernel_size': 1,
             'stride': 1, 'padding': 0, 'bias': True, 'normalization': 'none', 'activation': 'none',
             'num_mode': n_classes, 'controller_rate': controller_rate})
        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = make_model(
            {'cell': 'MCConv2dCell', 'input_size': input_dim, 'output_size': 2 * output_dim, 'kernel_size': kernel_shp,
             'stride': 1, 'padding': padding_shp, 'bias': True, 'normalization': 'none', 'activation': 'none',
             'num_mode': n_classes, 'controller_rate': controller_rate})
        self.horiz_resid = make_model(
            {'cell': 'MCConv2dCell', 'input_size': output_dim, 'output_size': output_dim, 'kernel_size': 1, 'stride': 1,
             'padding': 0, 'bias': True, 'normalization': 'bn', 'activation': 'none', 'num_mode': n_classes,
             'controller_rate': controller_rate})
        self.gate = GatedActivation()
        self.bn_v = nn.BatchNorm2d(output_dim)

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h):
        if self.mask_type == 'A':
            self.make_causal()
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert)
        out_v = self.bn_v(out_v)
        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)
        out = self.gate(v2h + h_horiz)
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)
        return out_v, out_h


class MCGatedPixelCNN(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=128, output_dim=256, n_layers=15, n_classes=10, controller_rate=0.5):
        super(MCGatedPixelCNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()
        # Initial block with Mask-A convolution
        self.layers.append(MCGatedMaskedConv2d('A', input_dim, hidden_dim, 7, False, n_classes, controller_rate))
        # Rest with Mask-B convolutions
        for i in range(n_layers - 1):
            self.layers.append(MCGatedMaskedConv2d('B', hidden_dim, hidden_dim, 3, True, n_classes, controller_rate))
        # Add the output layer
        self.output_conv = nn.Sequential(
            make_model({'cell': 'MCConv2dCell', 'input_size': hidden_dim, 'output_size': 512,
                        'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': True, 'normalization': 'none',
                        'activation': 'relu', 'num_mode': n_classes, 'controller_rate': controller_rate}),
            nn.Conv2d(512, input_dim * output_dim, 1)
        )

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['img']
        config.PARAM['indicator'] = F.one_hot(input['label'], config.PARAM['classes_size']).float()
        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h)
        x = self.output_conv(x_h)
        output['logits'] = x.view(x.size(0), self.output_dim, self.input_dim, x.size(2), x.size(3))
        output['loss'] = F.cross_entropy(output['logits'], ((input['img'] + 1) / 2 * 255).long().detach())
        return output

    def generate(self, x, C):
        input = {'img': x, 'label': C}
        for i in range(x.size(1)):
            for j in range(x.size(2)):
                output = self.forward(input)
                logits = output['logits'].permute(0, 2, 1, 3, 4)
                logits = logits.reshape(-1, logits.size(2), logits.size(3), logits.size(4))
                probs = F.softmax(logits[:, :, i, j], -1)
                samples = probs.multinomial(1).squeeze().view(input['img'].size(0), -1).float() / 255 * 2 - 1
                input['img'].data[:, :, i, j].copy_(samples)
        return input['img']


def cpixelcnn():
    input_size = config.PARAM['img_shape'][0]
    num_mode = config.PARAM['classes_size']
    n_layers = config.PARAM['n_layers']
    hidden_size = config.PARAM['hidden_size']
    model = ConditionalGatedPixelCNN(input_dim=input_size, hidden_dim=hidden_size, output_dim=256, n_layers=n_layers,
                                     n_classes=num_mode)
    return model


def mcpixelcnn():
    input_size = config.PARAM['img_shape'][0]
    num_mode = config.PARAM['classes_size']
    controller_rate = config.PARAM['controller_rate']
    n_layers = config.PARAM['n_layers']
    hidden_size = config.PARAM['hidden_size']
    model = MCGatedPixelCNN(input_dim=input_size, hidden_dim=hidden_size, output_dim=256, n_layers=n_layers,
                            n_classes=num_mode, controller_rate=controller_rate)
    return model