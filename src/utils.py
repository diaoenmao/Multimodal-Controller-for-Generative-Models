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


def save_img(img, path, nrow=10, padding=0, pad_value=0):
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
    config.PARAM['mode_data_size'] = int(config.PARAM['control']['mode_data_size'])
    if 'mode_param_rate' in config.PARAM['control']:
        config.PARAM['mode_param_rate'] = float(config.PARAM['control']['mode_param_rate'])
    if config.PARAM['data_name'] in ['MNIST', 'FashionMNIST']:
        config.PARAM['img_shape'] = [1, 32, 32]
        config.PARAM['generate_per_mode'] = 1000
    elif config.PARAM['data_name'] in ['Omniglot']:
        config.PARAM['img_shape'] = [1, 32, 32]
        config.PARAM['generate_per_mode'] = 20
    elif config.PARAM['data_name'] in ['SVHN', 'CIFAR10', 'CIFAR100']:
        config.PARAM['img_shape'] = [3, 32, 32]
        config.PARAM['generate_per_mode'] = 1000
    else:
        raise ValueError('Not valid dataset')
    # if config.PARAM['data_name'] in ['MNIST', 'FashionMNIST', 'EMNIST', 'Omniglot']:
    #     if config.PARAM['model_name'] in ['cgan', 'mcgan']:
    #         config.PARAM['latent_size'] = 100
    #         config.PARAM['hidden_size'] = 100
    #         config.PARAM['num_layers_generator'] = 5
    #         config.PARAM['num_layers_discriminator'] = 4
    #     elif config.PARAM['model_name'] in ['dccgan', 'dcmcgan']:
    #         config.PARAM['latent_size'] = 100
    #         config.PARAM['hidden_size'] = 50
    #         config.PARAM['depth'] = 3
    # else:
    #     raise ValueError('Not valid dataset')
    config.PARAM['embedding_size'] = 32
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
    if config.PARAM['subset'] == 'label':
        make_mode_dataset(dataset)
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



# def make_codebook(N, M, K):
#     search_range = 1000
#     codebook = set()
#     sum = np.zeros(N, dtype=np.int64)
#     for i in range(K):
#         sorted_seq = sum[::-1] if i == 0 else np.sort(sum)
#         sorted_ind = np.arange(N)[::-1] if i == 0 else np.argsort(sum)
#         seq_c = itertools.combinations(sorted_seq, M)
#         ind_c = itertools.combinations(sorted_ind, M)
#         p = 0
#         while True:
#             prev_size = len(codebook)
#             seq_s = np.array(list(itertools.islice(seq_c, p, p + search_range)))
#             ind_s = np.array(list(itertools.islice(ind_c, p, p + search_range)))
#             seq_sum = seq_s.sum(axis=1)
#             ind_s = ind_s[sorted(range(len(seq_sum)), key=lambda k: seq_sum[k])]
#             for idx in ind_s:
#                 code_c = np.zeros(N, dtype=np.int64)
#                 code_c[idx] = 1
#                 str_code = ''.join(str(cc) for cc in code_c.tolist())
#                 codebook.add(str_code)
#                 if len(codebook) > prev_size:
#                     sum += code_c
#                     break
#             if len(codebook) > prev_size:
#                 break
#             p += search_range
#     codebook = sorted(list(codebook))
#     for i in range(len(codebook)):
#         codebook[i] = [int(c) for c in codebook[i]]
#     codebook = np.array(codebook)
#     return codebook