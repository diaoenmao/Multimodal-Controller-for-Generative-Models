import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param


class GatedActivation(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.bn = nn.BatchNorm2d(hidden_size)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        x = self.bn(x)
        x = self.activation(x) * torch.sigmoid(y)
        return x


class ConditionalGatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, hidden_size, kernel, residual, num_mode):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual
        self.class_cond_embedding = nn.Embedding(num_mode, 2 * hidden_size)
        kernel_shp = (kernel // 2 + 1, kernel)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(hidden_size, 2 * hidden_size, kernel_shp, 1, padding_shp)
        self.vert_to_horiz = nn.Conv2d(2 * hidden_size, 2 * hidden_size, 1)
        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(hidden_size, 2 * hidden_size, kernel_shp, 1, padding_shp)
        self.gate_v = GatedActivation(hidden_size)
        self.gate_h = GatedActivation(hidden_size)
        self.horiz_resid = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 1),
            nn.BatchNorm2d(hidden_size),
        )

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()
        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate_v(h_vert + h[:, :, None, None])
        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)
        out_h = self.gate_h(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out_h) + x_h
        else:
            out_h = self.horiz_resid(out_h)
        return out_v, out_h


class ConditionalGatedPixelCNN(nn.Module):
    def __init__(self, input_size=256, hidden_size=64, num_layer=15, num_mode=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()
        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(num_layer):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True
            self.layers.append(ConditionalGatedMaskedConv2d(mask_type, hidden_size, kernel, residual, num_mode))
        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_size, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, input_size, 1)
        )

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        indicator = input['label']
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, indicator)
        output['logits'] = self.output_conv(x_h)
        output['loss'] = F.cross_entropy(output['logits'], input['img'])
        return output

    def generate(self, C, x=None):
        if x is None:
            x = torch.zeros((C.size(0), 8, 8), dtype=torch.long, device=cfg['device'])
        input = {'img': x, 'label': C}
        for i in range(x.size(1)):
            for j in range(x.size(2)):
                output = self.forward(input)
                probs = F.softmax(output['logits'][:, :, i, j], -1)
                input['img'].data[:, i, j].copy_(probs.multinomial(1).squeeze().data)
        return input['img']


def cpixelcnn():
    num_embedding = cfg['pixelcnn']['num_embedding']
    num_layer = cfg['pixelcnn']['num_layer']
    hidden_size = cfg['pixelcnn']['hidden_size']
    num_mode = cfg['classes_size']
    model = ConditionalGatedPixelCNN(input_size=num_embedding, hidden_size=hidden_size, num_layer=num_layer,
                                     num_mode=num_mode)
    model.apply(init_param)
    return model