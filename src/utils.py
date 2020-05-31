import collections.abc as container_abcs
import config
import errno
import numpy as np
import itertools
import os
import torch
from itertools import repeat
from torchvision.utils import save_image


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


def save_img(img, path, nrow=10, padding=2, pad_value=0):
    makedir_exist_ok(os.path.dirname(path))
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value)
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


def process_control_name():
    if 'controller_rate' in config.PARAM['control']:
        config.PARAM['controller_rate'] = float(config.PARAM['control']['controller_rate'])
    if config.PARAM['data_name'] in ['MNIST', 'FashionMNIST']:
        config.PARAM['img_shape'] = [1, 32, 32]
        config.PARAM['generate_per_mode'] = 1000
    elif config.PARAM['data_name'] in ['Omniglot']:
        config.PARAM['img_shape'] = [1, 32, 32]
        config.PARAM['generate_per_mode'] = 20
    elif config.PARAM['data_name'] in ['SVHN', 'CIFAR10', 'CIFAR100']:
        config.PARAM['img_shape'] = [3, 32, 32]
        config.PARAM['generate_per_mode'] = 1000
    elif config.PARAM['data_name'] in ['ImageNet32']:
        config.PARAM['img_shape'] = [3, 32, 32]
        config.PARAM['generate_per_mode'] = 20
    elif config.PARAM['data_name'] in ['ImageNet64']:
        config.PARAM['img_shape'] = [3, 64, 64]
        config.PARAM['generate_per_mode'] = 20
    else:
        raise ValueError('Not valid dataset')
    if config.PARAM['ae_name'] in ['vqvae']:
        config.PARAM['hidden_size'] = 128
        config.PARAM['conditional_embedding_size'] = 32
        config.PARAM['quantizer_embedding_size'] = 64
        config.PARAM['num_embedding'] = 512
        config.PARAM['vq_commit'] = 0.25
    if config.PARAM['model_name'] in ['cpixelcnn', 'mcpixelcnn']:
        config.PARAM['n_layers'] = 15
        config.PARAM['hidden_size'] = 128
        config.PARAM['num_embedding'] = 512
    elif config.PARAM['model_name'] in ['cvae', 'mcvae']:
        config.PARAM['hidden_size'] = [64, 128, 256]
        config.PARAM['latent_size'] = 128
        config.PARAM['conditional_embedding_size'] = 32
        config.PARAM['encode_shape'] = [config.PARAM['hidden_size'][-1],
                                        config.PARAM['img_shape'][1] // (2 ** 3),
                                        config.PARAM['img_shape'][2] // (2 ** 3)]
    elif config.PARAM['model_name'] in ['cgan', 'mcgan']:
        config.PARAM['generator_normalization'] = 'bn'
        config.PARAM['discriminator_normalization'] = 'none'
        config.PARAM['generator_activation'] = 'relu'
        config.PARAM['discriminator_activation'] = 'relu'
        config.PARAM['latent_size'] = 128
        config.PARAM['generator_hidden_size'] = [512, 256, 128, 64]
        config.PARAM['discriminator_hidden_size'] = [64, 128, 256, 512]
    elif config.PARAM['model_name'] in ['cglow', 'mcglow']:
        config.PARAM['hidden_size'] = 512
        config.PARAM['K'] = 16
        config.PARAM['L'] = 3
        config.PARAM['affine'] = True
        config.PARAM['conv_lu'] = True
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


def process_dataset(dataset):
    config.PARAM['classes_size'] = dataset.classes_size
    return


def make_mode_dataset(dataset):
    mode_img = []
    mode_target = []
    img = np.array(dataset.img)
    target = np.array(dataset.target[config.PARAM['subset']], dtype=np.int64)
    for i in range(config.PARAM['classes_size']):
        img_i = img[target == i]
        target_i = target[target == i]
        mode_data_size = len(target_i) if config.PARAM['mode_data_size'] == 0 else config.PARAM['mode_data_size']
        mode_img.append(img_i[:mode_data_size])
        mode_target.append(target_i[:mode_data_size])
    dataset.img = [img for model_img_i in mode_img for img in model_img_i]
    dataset.target[config.PARAM['subset']] = [target for model_target_i in mode_target for target in model_target_i]
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
        import datetime
        from logger import Logger
        last_epoch = 1
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/train_{}_{}'.format(config.PARAM['model_tag'], current_time) if config.PARAM[
            'log_overwrite'] else 'output/runs/train_{}'.format(config.PARAM['model_tag'])
        logger = Logger(logger_path)
    return last_epoch, model, optimizer, scheduler, logger


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input