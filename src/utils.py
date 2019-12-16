import collections.abc as container_abcs
import config
import errno
import numpy as np
import os
import torch
from itertools import repeat
from torchvision.utils import save_image


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
        np.save(path, input)
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'numpy':
        return np.load(path)
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path):
    makedir_exist_ok(os.path.dirname(path))
    save_image(img, path, padding=0)
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
    control_name = config.PARAM['control_name'].split('_')
    config.PARAM['channel_size'] = 3 if config.PARAM['data_name'] == 'CIFAR10' else 1
    config.PARAM['encoder_hidden_size'] = int(control_name[0])
    config.PARAM['encoder_sharing_rate']
    config.PARAM['embedding_size'] = int(control_name[1])
    config.PARAM['decoder_hidden_size'] = int(control_name[2])
    config.PARAM['decoder_sharing_rate']
    config.PARAM['scale_factor'] = 2
    config.PARAM['depth'] = int(control_name[3])
    config.PARAM['split_encoder'] = int(control_name[4])
    config.PARAM['split_mode_data'] = int(control_name[5])
    config.PARAM['split_mode_model'] = int(control_name[6])
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
    config.PARAM['stats'] = make_stats(dataset)
    config.PARAM['classes_size'] = dataset.classes_size
    if config.PARAM['split_mode_data'] == 0:
        label = torch.arange(config.PARAM['split_encoder']).repeat_interleave(len(dataset) // config.PARAM['split_encoder'] + 1)
        label = label[:len(dataset)]
        dataset.label = label.tolist()
    else:
        num_subset_class = config.PARAM['split_encoder'] // config.PARAM['classes_size']
        pivot = 0
        index = torch.arange(len(dataset.label))
        label = torch.tensor(dataset.label)
        new_label = label.clone()
        for i in range(config.PARAM['classes_size']):
            cur_index = torch.chunk(index[label == i], num_subset_class, dim=0)
            for j in range(len(cur_index)):
                new_label[cur_index[j]] = pivot
                pivot += 1
        dataset.label = new_label.tolist()
    return


def resume(model, model_tag, optimizer=None, scheduler=None, load_tag='checkpoint'):
    if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
        checkpoint = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
        logger = checkpoint['logger']
        print('Resume from {}'.format(last_epoch))
        return last_epoch, model, optimizer, scheduler, logger
    else:
        raise ValueError('Not exists model tag')
    return


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input