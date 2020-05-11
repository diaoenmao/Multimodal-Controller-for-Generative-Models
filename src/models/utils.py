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
    if config.PARAM['model_name'] in ['cgan', 'mcgan']:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    return m


def make_SpectralNormalization(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        return torch.nn.utils.spectral_norm(m)
    else:
        return m


def convex_combination(embedding):
    config.PARAM['concentration'] = torch.ones(embedding.size(0))
    m = torch.distributions.dirichlet.Dirichlet(config.PARAM['concentration'].to(config.PARAM['device']))
    cc = m.sample((config.PARAM['classes_size'],))
    cc_embedding = cc.matmul(embedding)
    return cc_embedding


def create_codebook(codebook):
    C, K = codebook.size(0), codebook.size(1)
    create_mode = config.PARAM['create_mode']
    classes_size = config.PARAM['classes_size']
    if create_mode == 'random':
        d = torch.distributions.bernoulli.Bernoulli(probs=0.5)
        created_codebook = set()
        while len(created_codebook) < classes_size:
            created_codebook_c = d.sample((classes_size, K))
            created_codebook_c = [tuple(c) for c in created_codebook_c.tolist()]
            created_codebook.update(created_codebook_c)
    else:
        mutate_threshold = 0.5
        created_codebook = set()
        while len(created_codebook) < classes_size:
            selected_parents = torch.randint(C, (2,))
            parents = codebook[selected_parents]
            if create_mode == 'single':
                crossover_point = torch.randint(K, (1,)).to(config.PARAM['device'])
                created_codebook_c = torch.cat([parents[0, :crossover_point], parents[1, crossover_point:]])
            elif create_mode == 'half':
                crossover_point = K // 2
                created_codebook_c = torch.cat([parents[0, :crossover_point], parents[1, crossover_point:]])
            elif create_mode == 'uniform':
                crossover_point = torch.randint(2, (K,)).to(config.PARAM['device'])
                created_codebook_c = parents.gather(0, crossover_point.view(1, -1)).squeeze()
            else:
                raise ValueError('Not valid crossover mode')
            mutate_mask = torch.rand(K) < mutate_threshold
            created_codebook_c[mutate_mask] = torch.fmod(created_codebook_c[mutate_mask] + 1, 2)
            created_codebook_c = tuple(created_codebook_c.tolist())
            created_codebook.add(created_codebook_c)
    created_codebook = torch.tensor(list(created_codebook)[:classes_size], dtype=torch.float)
    return created_codebook


def create(model):
    if 'VAE' in model.__class__.__name__:
        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name == 'MultimodalController':
                new_codebook = create_codebook(module.codebook)
                module.register_buffer('codebook', new_codebook)
            if isinstance(module, nn.Linear):
                name_list = name.split('.')
                if len(name_list) >= 2 and 'embedding' in name_list[1]:
                    module.weight = nn.Parameter(convex_combination(module.weight.t()).t())
    elif 'PixelCNN' in model.__class__.__name__:
        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name == 'MultimodalController':
                new_codebook = create_codebook(module.codebook)
                module.register_buffer('codebook', new_codebook)
            if isinstance(module, nn.Embedding):
                name_list = name.split('.')
                if len(name_list) >= 2 and 'class_cond_embedding' in name_list[2]:
                    module.weight = nn.Parameter(convex_combination(module.weight))
    elif 'GAN' in model.__class__.__name__:
        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name == 'MultimodalController':
                new_codebook = create_codebook(module.codebook)
                module.register_buffer('codebook', new_codebook)
            if isinstance(module, nn.Linear):
                name_list = name.split('.')
                if len(name_list) >= 2 and 'embedding' in name_list[1]:
                    module.weight = nn.Parameter(convex_combination(module.weight.t()).t())
    return