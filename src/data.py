import config
import torch
import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def fetch_dataset(data_name, subset):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['MNIST', 'FashionMNIST', 'SVHN']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', subset=subset,'
                                'transform=datasets.Compose([''transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', subset=subset,'
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        config.PARAM['transform'] = {
            'train': datasets.Compose([transforms.ToTensor()]), 'test': datasets.Compose([transforms.ToTensor()])
        }
    elif data_name == 'EMNIST':
        dataset['train'] = datasets.EMNIST(root=root, split='train', subset=subset,
                                           transform=datasets.Compose([transforms.ToTensor()]))
        dataset['test'] = datasets.EMNIST(root=root, split='test', subset=subset,
                                          transform=datasets.Compose([transforms.ToTensor()]))
        config.PARAM['transform'] = {
            'train': datasets.Compose([transforms.ToTensor()]), 'test': datasets.Compose([transforms.ToTensor()])
        }
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', subset=subset,'
                                'transform=datasets.Compose([''transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', subset=subset,'
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        config.PARAM['transform'] = {
            'train': datasets.Compose([transforms.ToTensor()]),
            'test': datasets.Compose([transforms.ToTensor()])
        }
    elif data_name == 'ImageNet':
        dataset['train'] = datasets.ImageNet(root, split='train', subset=subset,
                                             transform=datasets.Compose([transforms.ToTensor()]))
        dataset['test'] = datasets.ImageNet(root, split='test', subset=subset,
                                            transform=datasets.Compose([transforms.ToTensor()]))
        config.PARAM['transform'] = {
            'train': datasets.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
            'test': datasets.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        }
    elif data_name == 'Kodak':
        dataset['train'] = datasets.ImageFolder(root, transform=datasets.Compose([transforms.ToTensor()]))
        dataset['test'] = datasets.ImageFolder(root, transform=datasets.Compose([transforms.ToTensor()]))
        config.PARAM['transform'] = {
            'train': datasets.Compose([transforms.ToTensor()]), 'test': datasets.Compose([transforms.ToTensor()])
        }
    elif data_name == 'Omniglot':
        dataset['train'] = datasets.Omniglot(root=root, split='train', subset=subset,
                                             transform=datasets.Compose([transforms.ToTensor()]))
        dataset['test'] = datasets.Omniglot(root=root, split='test', subset=subset,
                                            transform=datasets.Compose([transforms.ToTensor()]))
        config.PARAM['transform'] = {
            'train': datasets.Compose([transforms.Resize((28, 28)), transforms.ToTensor()]),
            'test': datasets.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
        }
    elif data_name == 'CUB200':
        dataset['train'] = datasets.CUB200(root=root, split='train', subset=subset,
                                           transform=datasets.Compose([transforms.ToTensor()]))
        dataset['test'] = datasets.CUB200(root=root, split='test', subset=subset,
                                          transform=datasets.Compose([transforms.ToTensor()]))
        config.PARAM['transform'] = {
            'train': datasets.Compose([datasets.transforms.BoundingBoxCrop(), transforms.Resize((64, 64)),
                                       transforms.ToTensor()]),
            'test': datasets.Compose([datasets.transforms.BoundingBoxCrop(), transforms.Resize((64, 64)),
                                      transforms.ToTensor()])
        }
    elif data_name == 'CelebA':
        dataset['train'] = datasets.CelebA(root=root, split='train', subset=subset,
                                           transform=datasets.Compose([transforms.ToTensor()]))
        dataset['test'] = datasets.CelebA(root=root, split='test', subset=subset,
                                          transform=datasets.Compose([transforms.ToTensor()]))
        config.PARAM['transform'] = {
            'train': datasets.Compose([transforms.Resize((78, 64)), transforms.ToTensor()]),
            'test': datasets.Compose([transforms.Resize((78, 64)), transforms.ToTensor()])
        }
    else:
        raise ValueError('Not valid dataset name')
    dataset['train'].transform = config.PARAM['transform']['train']
    dataset['test'].transform = config.PARAM['transform']['test']
    print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset):
    data_loader = {}
    for k in dataset:
        data_loader[k] = torch.utils.data.DataLoader(dataset=dataset[k], shuffle=config.PARAM['shuffle'][k],
                                                     batch_size=config.PARAM['batch_size'][k], pin_memory=True,
                                                     num_workers=config.PARAM['num_workers'], collate_fn=input_collate)
    return data_loader