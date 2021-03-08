import anytree
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, save, load
from .utils import make_classes_counts, make_tree, make_flat_index


class CelebAHQ(Dataset):
    data_name = 'CelebA-HQ'

    def __init__(self, root, split, subset, size, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.subset = subset
        self.size = size
        self.transform = transform
        if not check_exists(os.path.join(self.processed_folder, str(self.size))):
            self.process()
        self.img, self.target = load(os.path.join(self.processed_folder, str(self.size), '{}.pt'.format(self.split)))
        self.target = self.target[self.subset]
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.classes_size = load(os.path.join(self.processed_folder, str(self.size), 'meta.pt'))
        self.classes_to_labels, self.classes_size = self.classes_to_labels[self.subset], self.classes_size[self.subset]

    def __getitem__(self, index):
        img, target = Image.open(self.img[index], mode='r').convert('RGB'), torch.tensor(self.target[index])
        input = {'img': img, self.subset: target}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.img)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            raise ValueError('Dataset not found')
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, str(self.size), 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, str(self.size), 'test.pt'))
        save(meta, os.path.join(self.processed_folder, str(self.size), 'meta.pt'))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nSubset: {}\nSize: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.subset, self.size,
            self.transform.__repr__())
        return fmt_str

    def make_data(self):
        attr = pd.read_csv(os.path.join(self.raw_folder, 'Anno', 'list_attr_celeba.txt'), delim_whitespace=True,
                           header=1)
        attr_name = attr.columns.tolist()
        label = pd.read_csv(os.path.join(self.raw_folder, 'Anno', 'identity_CelebA.txt'), delim_whitespace=True,
                            names=['label'], index_col=0)
        filename = os.listdir(os.path.join(self.raw_folder, 'celeba-hq', 'celeba-{}'.format(self.size)))
        img = [os.path.join(self.raw_folder, 'celeba-hq', 'celeba-{}'.format(self.size), x) for x in filename]
        attr = ((attr.loc[filename, :].values + 1) // 2).astype(np.int64)
        label = label.loc[filename, 'label'].values
        unique_label = np.unique(label).tolist()
        label = np.array(list(map(lambda x: unique_label.index(x), label))).astype(np.int64)
        label_name = np.sort(np.unique(label)).astype(str).tolist()
        train_img, test_img = img, img
        train_attr, test_attr = attr, attr
        train_label, test_label = label, label
        train_target = {'attr': train_attr, 'label': train_label}
        test_target = {'attr': test_attr, 'label': test_label}
        classes_to_labels = {'attr': anytree.Node('U', index=[]), 'label': anytree.Node('U', index=[])}
        for a in attr_name:
            make_tree(classes_to_labels['attr'], [a])
        for i in label_name:
            make_tree(classes_to_labels['label'], [i])
        classes_size = {'attr': make_flat_index(classes_to_labels['attr']),
                        'label': make_flat_index(classes_to_labels['label'])}
        return (train_img, train_target), (test_img, test_target), (classes_to_labels, classes_size)