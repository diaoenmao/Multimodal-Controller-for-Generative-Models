import collections.abc as container_abcs
import errno
import numpy as np
import os
import torch
from itertools import repeat
from torchvision.utils import save_image
from config import cfg


def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, protocol=2, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path, pickle_protocol=protocol)
    elif mode == 'numpy':
        np.save(path, input, allow_pickle=True)
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'numpy':
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path, nrow=10, padding=2, pad_value=0, range=None):
    makedir_exist_ok(os.path.dirname(path))
    normalize = False if range is None else True
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    else:
        raise ValueError('Not valid input type')
    return output


def process_dataset(dataset):
    cfg['classes_size'] = dataset.classes_size
    return


def process_control():
    if 'controller_rate' in cfg['control']:
        cfg['controller_rate'] = float(cfg['control']['controller_rate'])
    if cfg['data_name'] in ['MNIST', 'FashionMNIST']:
        cfg['data_shape'] = [1, 32, 32]
        cfg['generate_per_mode'] = 1000
    elif cfg['data_name'] in ['Omniglot']:
        cfg['data_shape'] = [1, 32, 32]
        cfg['generate_per_mode'] = 20
    elif cfg['data_name'] in ['SVHN', 'CIFAR10', 'Dogs', 'CUB200', 'Cars']:
        cfg['data_shape'] = [3, 32, 32]
        cfg['generate_per_mode'] = 1000
    elif cfg['data_name'] in ['CIFAR100']:
        cfg['data_shape'] = [3, 32, 32]
        cfg['generate_per_mode'] = 100
    elif cfg['data_name'] in ['ImageNet32']:
        cfg['data_shape'] = [3, 32, 32]
        cfg['generate_per_mode'] = 20
    elif cfg['data_name'] in ['CelebA-HQ', 'ImageNet']:
        cfg['data_shape'] = [3, 128, 128]
        cfg['generate_per_mode'] = 20
    else:
        raise ValueError('Not valid dataset')
    if cfg['ae_name'] in ['vqvae']:
        cfg['vqvae'] = {}
        if cfg['data_shape'][1] == 32:
            cfg['vqvae']['hidden_size'] = [128, 128]
        elif cfg['data_shape'][1] == 128:
            cfg['vqvae']['hidden_size'] = [128, 128, 128, 128]
        else:
            raise ValueError('Not valid data shape')
        cfg['vqvae']['num_res_block'] = 2
        cfg['vqvae']['embedding_size'] = 64
        cfg['vqvae']['num_embedding'] = 512
        cfg['vqvae']['vq_commit'] = 0.25
    if cfg['model_name'] in ['cpixelcnn', 'mcpixelcnn']:
        cfg['pixelcnn'] = {}
        cfg['pixelcnn']['num_layer'] = 15
        cfg['pixelcnn']['hidden_size'] = 128
        cfg['pixelcnn']['num_embedding'] = 512
    elif cfg['model_name'] in ['cvae', 'mcvae']:
        cfg['vae'] = {}
        if cfg['data_shape'][1] == 32:
            cfg['vae']['hidden_size'] = [64, 128, 256]
            cfg['vae']['latent_size'] = 128
        elif cfg['data_shape'][1] == 128:
            cfg['vae']['hidden_size'] = [64, 128, 256, 512, 512]
            cfg['vae']['latent_size'] = 256
        else:
            raise ValueError('Not valid data shape')
        cfg['vae']['num_res_block'] = 2
        cfg['vae']['embedding_size'] = 32
    elif cfg['model_name'] in ['cgan', 'mcgan']:
        cfg['gan'] = {}
        cfg['gan']['latent_size'] = 128
        if cfg['data_shape'][1] == 32:
            if cfg['data_name'] in ['CIFAR10', 'CIFAR100']:
                cfg['gan']['generator_hidden_size'] = [256, 256, 256, 256]
                cfg['gan']['discriminator_hidden_size'] = [128, 128, 128, 128]
            else:
                cfg['gan']['generator_hidden_size'] = [512, 256, 128, 64]
                cfg['gan']['discriminator_hidden_size'] = [64, 128, 256, 512]
        elif cfg['data_shape'][1] == 128:
            cfg['gan']['generator_hidden_size'] = [1024, 512, 256, 128, 64]
            cfg['gan']['discriminator_hidden_size'] = [64, 128, 256, 512, 1024, 1024]
        else:
            raise ValueError('Not valid data shape')
        cfg['gan']['embedding_size'] = 32
    elif cfg['model_name'] in ['cglow', 'mcglow']:
        cfg['glow'] = {}
        cfg['glow']['hidden_size'] = 512
        if cfg['data_shape'][1] == 32:
            cfg['glow']['K'] = 16
            cfg['glow']['L'] = 3
        elif cfg['data_shape'][1] == 128:
            cfg['glow']['K'] = 16
            cfg['glow']['L'] = 5
        else:
            raise ValueError('Not valid data shape')
        cfg['glow']['affine'] = True
        cfg['glow']['conv_lu'] = True
    cfg['classifier'] = {}
    cfg['classifier']['hidden_size'] = [8, 16, 32, 64]
    return


def make_stats(dataset):
    if os.path.exists('./data/stats/{}.pt'.format(dataset.data_name)):
        stats = load('./data/stats/{}.pt'.format(dataset.data_name))
    elif dataset is not None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
        stats = Stats(dim=1)
        with torch.no_grad():
            for input in data_loader:
                stats.update(input['img'])
        save(stats, './data/stats/{}.pt'.format(dataset.data_name))
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)
            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n
        return


def resume(model, model_tag, optimizer=None, scheduler=None, load_tag='checkpoint', verbose=True):
    if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
        checkpoint = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
        logger = checkpoint['logger']
        if verbose:
            print('Resume from {}'.format(last_epoch))
    else:
        print('Not exists model tag: {}, start from scratch'.format(model_tag))
        from datetime import datetime
        from logger import Logger
        last_epoch = 1
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
    return last_epoch, model, optimizer, scheduler, logger


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input