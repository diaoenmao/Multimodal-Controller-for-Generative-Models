import config
import torch
import torch.nn as nn
from modules import make_cell


def make_model(model):
    if isinstance(model, dict):
        if 'cell' in model:
            return make_cell(model)
        elif 'nn' in model:
            return eval(model['nn'])
        else:
            cell = nn.ModuleDict({})
            for k in model:
                cell[k] = make_model(model[k])
            return cell
    elif isinstance(model, list):
        cell = nn.ModuleList([])
        for i in range(len(model)):
            cell.append(make_model(model[i]))
        return cell
    elif isinstance(model, tuple):
        container = []
        for i in range(len(model)):
            container.append(make_model(model[i]))
        cell = nn.Sequential(*container)
        return cell
    else:
        raise ValueError('Not valid model format')
    return


def normalize(input):
    broadcast_size = [1] * input.dim()
    broadcast_size[1] = input.size(1)
    m = config.PARAM['stats'].mean.view(broadcast_size).to(input.device)
    s = config.PARAM['stats'].std.view(broadcast_size).to(input.device)
    input = input.sub(m).div(s)
    return input


def denormalize(input):
    broadcast_size = [1] * input.dim()
    broadcast_size[1] = input.size(1)
    m = config.PARAM['stats'].mean.view(broadcast_size).to(input.device)
    s = config.PARAM['stats'].std.view(broadcast_size).to(input.device)
    input = input.mul(s).add(m)
    return input


def init_param(m):
    if isinstance(m, (nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    return m


def convex_combination(embedding):
    config.PARAM['concentration'] = torch.ones(config.PARAM['classes_size'])
    m = torch.distributions.dirichlet.Dirichlet(config.PARAM['concentration'].to(config.PARAM['device']))
    cc = m.sample((config.PARAM['classes_size'],))
    cc_embedding = cc.matmul(embedding)
    return cc_embedding


def create(model):
    if 'VAE' in model.__class__.__name__:
        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name == 'MultimodalController':
                module.register_buffer('codebook', module.make_codebook())
                # module.codebook.data.copy_(covex_combination(module.codebook).data)
            if isinstance(module, nn.Linear):
                name_list = name.split('.')
                if len(name_list) >= 2 and 'embedding' in name_list[1]:
                    module.weight.data.copy_(convex_combination(module.weight.t()).t().data)
    elif 'PixelCNN' in model.__class__.__name__:
        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name == 'MultimodalController':
                module.register_buffer('codebook', module.make_codebook())
                # module.codebook.data.copy_(convex_combination(module.codebook).data)
            if isinstance(module, nn.Embedding):
                name_list = name.split('.')
                if len(name_list) >= 2 and 'class_cond_embedding' in name_list[1]:
                    module.weight.data.copy_(convex_combination(module.weight.t()).t().data)
    return


def make_SpectralNormalization(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        return torch.nn.utils.spectral_norm(m)
    else:
        return m