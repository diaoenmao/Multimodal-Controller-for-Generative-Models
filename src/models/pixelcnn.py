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
    def __init__(self, mask_type, dim, kernel, residual, n_classes):
        super(ConditionalGatedMaskedConv2d, self).__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual
        self.class_cond_embedding = nn.Embedding(n_classes, 2 * dim)
        kernel_shp = (kernel // 2 + 1, kernel)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(dim, dim * 2, kernel_shp, 1, padding_shp)
        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)
        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(dim, dim * 2, kernel_shp, 1, padding_shp)
        self.horiz_resid = nn.Conv2d(dim, dim, 1)
        self.gate = GatedActivation()

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
        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)
        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)
        return out_v, out_h


class ConditionalGatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
        super(ConditionalGatedPixelCNN, self).__init__()
        self.input_dim = input_dim
        self.dim = dim
        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)
        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()
        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True
            self.layers.append(ConditionalGatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes))
        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['code']
        label = input['label']
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, W, W)
        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)
        output['logits'] = self.output_conv(x_h)
        output['loss'] = F.cross_entropy(output['logits'], input['code'])
        return output

    def generate(self, x, C):
        input = {'code': x, 'label': C}
        for i in range(x.size(1)):
            for j in range(x.size(2)):
                output = self.forward(input)
                probs = F.softmax(output['logits'][:, :, i, j], -1)
                input['code'].data[:, i, j].copy_(probs.multinomial(1).squeeze().data)
        return input['code']


class MCGatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual, n_classes, controller_rate):
        super(MCGatedMaskedConv2d, self).__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual
        kernel_shp = (kernel // 2 + 1, kernel)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = make_model(
            {'cell': 'MCConv2dCell', 'input_size': dim, 'output_size': dim * 2, 'kernel_size': kernel_shp, 'stride': 1,
             'padding': padding_shp, 'bias': True, 'normalization': 'none', 'activation': 'none', 'num_mode': n_classes,
             'controller_rate': controller_rate})
        self.vert_to_horiz = make_model(
            {'cell': 'MCConv2dCell', 'input_size': dim * 2, 'output_size': dim * 2, 'kernel_size': 1,
             'stride': 1, 'padding': 0, 'bias': True, 'normalization': 'none', 'activation': 'none',
             'num_mode': n_classes, 'controller_rate': controller_rate})
        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = make_model(
            {'cell': 'MCConv2dCell', 'input_size': dim, 'output_size': dim * 2, 'kernel_size': kernel_shp, 'stride': 1,
             'padding': padding_shp, 'bias': True, 'normalization': 'none', 'activation': 'none', 'num_mode': n_classes,
             'controller_rate': controller_rate})
        self.horiz_resid = make_model(
            {'cell': 'MCConv2dCell', 'input_size': dim, 'output_size': dim, 'kernel_size': 1, 'stride': 1,
             'padding': 0, 'bias': True, 'normalization': 'none', 'activation': 'none', 'num_mode': n_classes,
             'controller_rate': controller_rate})
        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h):
        if self.mask_type == 'A':
            self.make_causal()
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert)
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
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, controller_rate=0.5):
        super(MCGatedPixelCNN, self).__init__()
        self.input_dim = input_dim
        self.dim = dim
        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)
        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()
        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True
            self.layers.append(MCGatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes, controller_rate))
        # Add the output layer
        self.output_conv = nn.Sequential(
            make_model({'cell': 'MCConv2dCell', 'input_size': dim, 'output_size': 512,
                        'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': True, 'normalization': 'none',
                        'activation': 'relu', 'num_mode': n_classes, 'controller_rate': controller_rate}),
            nn.Conv2d(512, input_dim, 1)
        )

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['code']
        config.PARAM['indicator'] = F.one_hot(input['label'], config.PARAM['classes_size']).float()
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, W, W)
        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h)
        output['logits'] = self.output_conv(x_h)
        output['loss'] = F.cross_entropy(output['logits'], input['code'])
        return output

    def generate(self, x, C):
        input = {'code': x, 'label': C}
        for i in range(x.size(1)):
            for j in range(x.size(2)):
                output = self.forward(input)
                probs = F.softmax(output['logits'][:, :, i, j], -1)
                input['code'].data[:, i, j].copy_(probs.multinomial(1).squeeze().data)
        return input['code']


def cgatedpixelcnn():
    num_mode = config.PARAM['classes_size']
    num_embedding = config.PARAM['num_embedding']
    n_layers = 15
    hidden_size = 64
    model = ConditionalGatedPixelCNN(input_dim=num_embedding, dim=hidden_size, n_layers=n_layers, n_classes=num_mode)
    return model


def mcgatedpixelcnn():
    num_mode = config.PARAM['classes_size']
    num_embedding = config.PARAM['num_embedding']
    controller_rate = config.PARAM['controller_rate']
    n_layers = 15
    hidden_size = 64
    model = MCGatedPixelCNN(input_dim=num_embedding, dim=hidden_size, n_layers=n_layers, n_classes=num_mode,
                            controller_rate=controller_rate)
    return model