import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import make_model


class GatedActivation(nn.Module):
    def __init__(self):
        super(GatedActivation, self).__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)


class MCGatedActivation(nn.Module):
    def __init__(self, dim, num_mode, controller_rate):
        super(MCGatedActivation, self).__init__()
        self.mc = make_model({'cell': 'MultimodalController', 'input_size': dim, 'num_mode': num_mode,
                              'controller_rate': controller_rate})

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return self.mc(torch.tanh(x)) * torch.sigmoid(y)


class ConditionalGatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual, n_classes):
        super(ConditionalGatedMaskedConv2d, self).__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual
        self.class_cond_embedding = nn.Embedding(n_classes, 2 * dim)
        kernel_shp = (kernel // 2 + 1, kernel)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(dim, 2 * dim, kernel_shp, 1, padding_shp)
        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)
        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(dim, 2 * dim, kernel_shp, 1, padding_shp)
        self.horiz_resid = nn.Conv2d(dim, dim, 1)
        self.gate = GatedActivation()
        self.bn_1 = nn.BatchNorm2d(dim)
        self.bn_2 = nn.BatchNorm2d(dim)

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
        out_v = self.bn_1(out_v)
        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)
        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.bn_2(self.horiz_resid(out)) + x_h
        else:
            out_h = self.bn_2(self.horiz_resid(out))
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
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        label = input['label']
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, W, W)
        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)
        output['logits'] = self.output_conv(x_h)
        output['loss'] = F.cross_entropy(output['logits'], input['img'])
        return output

    def generate(self, C, x=None):
        if x is None:
            x = torch.zeros((C.size(0), cfg['img_shape'][1] // 4, cfg['img_shape'][2] // 4),
                            dtype=torch.long, device=cfg['device'])
        input = {'img': x, 'label': C}
        for i in range(x.size(1)):
            for j in range(x.size(2)):
                output = self.forward(input)
                probs = F.softmax(output['logits'][:, :, i, j], -1)
                input['img'].data[:, i, j].copy_(probs.multinomial(1).squeeze().data)
        return input['img']


class MCGatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual, n_classes, controller_rate):
        super(MCGatedMaskedConv2d, self).__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual
        kernel_shp = (kernel // 2 + 1, kernel)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(dim, 2 * dim, kernel_shp, 1, padding_shp)
        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)
        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(dim, 2 * dim, kernel_shp, 1, padding_shp)
        self.horiz_resid = make_model(
            {'cell': 'MCConv2dCell', 'input_size': dim, 'output_size': dim, 'kernel_size': 1, 'stride': 1,
             'padding': 0, 'bias': True, 'normalization': 'bn', 'activation': 'none', 'num_mode': n_classes,
             'controller_rate': controller_rate})
        self.mcgate_1 = MCGatedActivation(dim, n_classes, controller_rate)
        self.mcgate_2 = MCGatedActivation(dim, n_classes, controller_rate)
        self.bn_1 = nn.BatchNorm2d(dim)

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h):
        if self.mask_type == 'A':
            self.make_causal()
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.mcgate_1(h_vert)
        out_v = self.bn_1(out_v)
        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)
        out = self.mcgate_2(v2h + h_horiz)
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
                        'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': True, 'normalization': 'bn',
                        'activation': 'relu', 'num_mode': n_classes, 'controller_rate': controller_rate}),
            nn.Conv2d(512, input_dim, 1)
        )

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        cfg['indicator'] = F.one_hot(input['label'], cfg['classes_size']).float()
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, W, W)
        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h)
        output['logits'] = self.output_conv(x_h)
        output['loss'] = F.cross_entropy(output['logits'], input['img'])
        return output

    def generate(self, C, x=None):
        if x is None:
            x = torch.zeros((C.size(0), cfg['img_shape'][1] // 4, cfg['img_shape'][2] // 4),
                            dtype=torch.long, device=cfg['device'])
        input = {'img': x, 'label': C}
        for i in range(x.size(1)):
            for j in range(x.size(2)):
                output = self.forward(input)
                probs = F.softmax(output['logits'][:, :, i, j], -1)
                input['img'].data[:, i, j].copy_(probs.multinomial(1).squeeze().data)
        return input['img']


def cpixelcnn():
    num_mode = cfg['classes_size']
    num_embedding = cfg['num_embedding']
    n_layers = cfg['n_layers']
    hidden_size = cfg['hidden_size']
    model = ConditionalGatedPixelCNN(input_dim=num_embedding, dim=hidden_size, n_layers=n_layers, n_classes=num_mode)
    return model


def mcpixelcnn():
    num_mode = cfg['classes_size']
    num_embedding = cfg['num_embedding']
    controller_rate = cfg['controller_rate']
    n_layers = cfg['n_layers']
    hidden_size = cfg['hidden_size']
    model = MCGatedPixelCNN(input_dim=num_embedding, dim=hidden_size, n_layers=n_layers, n_classes=num_mode,
                            controller_rate=controller_rate)
    return model