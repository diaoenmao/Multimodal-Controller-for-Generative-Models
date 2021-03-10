import torch
import torch.nn as nn
import numpy as np
from config import cfg


def init_param(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    if cfg['model_name'] in ['cgan', 'mcgan']:
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight.data, 1.)
    return m


def make_SpectralNormalization(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        return torch.nn.utils.spectral_norm(m)
    else:
        return m


def create_embedding(embedding):
    C, K = embedding.size(0), embedding.size(1)
    classes_size = cfg['classes_size']
    concentration = torch.ones(C, device=embedding.device)
    m = torch.distributions.dirichlet.Dirichlet(concentration)
    convex_combination = m.sample((classes_size,))
    created_embedding = (convex_combination.matmul(embedding)).to(cfg['device'])
    return created_embedding


def create_codebook(codebook):
    C, K = codebook.size(0), codebook.size(1)
    classes_size = cfg['classes_size']
    d = torch.distributions.bernoulli.Bernoulli(probs=0.5)
    created_codebook = set()
    while len(created_codebook) < classes_size:
        created_codebook_c = d.sample((classes_size, K))
        created_codebook_c = [tuple(c) for c in created_codebook_c.tolist()]
        created_codebook.update(created_codebook_c)
    created_codebook = torch.tensor(list(created_codebook)[:classes_size], dtype=torch.float).to(cfg['device'])
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
                if len(name_list) == 2 and 'embedding' in name_list[1]:
                    module.weight = nn.Parameter(create_embedding(module.weight.t()).t())
    elif 'PixelCNN' in model.__class__.__name__:
        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name == 'MultimodalController':
                new_codebook = create_codebook(module.codebook)
                module.register_buffer('codebook', new_codebook)
            if isinstance(module, nn.Embedding):
                name_list = name.split('.')
                if len(name_list) >= 3 and 'class_cond_embedding' in name_list[2]:
                    module.weight = nn.Parameter(create_embedding(module.weight))
    elif 'Glow' in model.__class__.__name__:
        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name == 'MultimodalController':
                new_codebook = create_codebook(module.codebook)
                module.register_buffer('codebook', new_codebook)
            name_list = name.split('.')
            if len(name_list) == 4 and 'embedding' in name_list[2]:
                module.weight = nn.Parameter(
                    create_embedding(module.weight.squeeze().t()).t().unsqueeze(2).unsqueeze(3))
    elif 'GAN' in model.__class__.__name__:
        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name == 'MultimodalController':
                new_codebook = create_codebook(module.codebook)
                module.register_buffer('codebook', new_codebook)
            if isinstance(module, nn.Linear):
                name_list = name.split('.')
                if len(name_list) >= 2 and 'embedding' in name_list[1]:
                    module.weight = nn.Parameter(create_embedding(module.weight.t()).t())
    return


def transit_embedding(embedding, root, alpha):
    embedding = embedding.cpu().numpy()
    root_embedding = embedding[root]
    transited_embedding = np.delete(embedding, root, 0)
    transited_embedding = alpha * transited_embedding + (1 - alpha) * root_embedding
    transited_embedding = np.insert(transited_embedding, root, root_embedding, 0)
    transited_embedding = torch.tensor(transited_embedding, device=cfg['device'])
    return transited_embedding


def transit_codebook(codebook, root, alpha):
    codebook = codebook.cpu().numpy()
    root_code = codebook[root]
    transited_codebook = np.delete(codebook, root, 0)
    cross_point = int(round((1 - alpha) * codebook.shape[1]))
    transited_codebook[:, :cross_point] = root_code[:cross_point]
    transited_codebook = np.insert(transited_codebook, root, root_code, 0)
    transited_codebook = torch.tensor(transited_codebook, device=cfg['device'])
    return transited_codebook


def transit(model, root, alpha):
    if 'VAE' in model.__class__.__name__:
        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name == 'MultimodalController':
                if not hasattr(module, 'codebook_orig'):
                    module.register_buffer('codebook_orig', module.codebook.data)
                module.register_buffer('codebook', transit_codebook(module.codebook_orig, root, alpha))
            if isinstance(module, nn.Linear):
                name_list = name.split('.')
                if len(name_list) == 2 and 'embedding' in name_list[1]:
                    if not hasattr(module, 'weight_orig'):
                        module.register_buffer('weight_orig', module.weight.data)
                    module.weight = nn.Parameter(transit_embedding(module.weight_orig.t(), root, alpha).t())
    elif 'Glow' in model.__class__.__name__:
        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name == 'MultimodalController':
                if not hasattr(module, 'codebook_orig'):
                    module.register_buffer('codebook_orig', module.codebook.data)
                module.register_buffer('codebook', transit_codebook(module.codebook_orig, root, alpha))
            name_list = name.split('.')
            if len(name_list) == 4 and 'embedding' in name_list[2]:
                if not hasattr(module, 'weight_orig'):
                    module.register_buffer('weight_orig', module.weight.data)
                module.weight = nn.Parameter(
                    transit_embedding(module.weight_orig.squeeze().t(), root, alpha).t().unsqueeze(2).unsqueeze(3))
    elif 'GAN' in model.__class__.__name__:
        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name == 'MultimodalController':
                if not hasattr(module, 'codebook_orig'):
                    module.register_buffer('codebook_orig', module.codebook.data)
                module.register_buffer('codebook', transit_codebook(module.codebook_orig, root, alpha))
            if isinstance(module, nn.Linear):
                name_list = name.split('.')
                if len(name_list) >= 2 and 'embedding' in name_list[1]:
                    if not hasattr(module, 'weight_orig'):
                        module.register_buffer('weight_orig', module.weight.data)
                    module.weight = nn.Parameter(transit_embedding(module.weight_orig.t(), root, alpha).t())
    return
