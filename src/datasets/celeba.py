import anytree
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import makedir_exist_ok, save, load
from .utils import download_google, extract_file, make_classes_counts, make_tree, make_flat_index


class CelebA(Dataset):
    data_name = 'CelebA'
    file = [
        #(ID, MD5, Filename)
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(self, root, split, subset, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.subset = subset
        self.transform = transform
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
        self.img, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))
        self.classes_to_labels, self.classes_size = load(os.path.join(self.processed_folder, 'meta.pt'))
        self.target = self.target[self.subset]
        if self.subset in ['attr', 'identity']:
            self.classes_to_labels = self.classes_to_labels[self.subset]
            self.classes_size = self.classes_size[self.subset]
            if self.subset == 'identity':
                self.classes_counts = make_classes_counts(self.target)

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

    def _check_exists(self):
        return os.path.exists(self.processed_folder)

    def download(self):
        if self._check_exists():
            return
        makedir_exist_ok(self.raw_folder)
        for (id, md5, filename) in self.file:
            download_google(id, os.path.join(self.raw_folder, filename))
        extract_file(os.path.join(self.raw_folder, 'img_align_celeba.zip'))
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\n'.format(self.__class__.__name__)
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root: {}\n'.format(self.root)
        fmt_str += '    Transforms: {}\n'.format(self.transform.__repr__())
        return fmt_str

    def make_data(self):
        attr = pd.read_csv(os.path.join(self.raw_folder, 'list_attr_celeba.txt'), delim_whitespace=True, header=1)
        identity = pd.read_csv(os.path.join(self.raw_folder, 'identity_CelebA.txt'), delim_whitespace=True,
                               names=['identity'], index_col=0)
        bbox = pd.read_csv(os.path.join(self.raw_folder, 'list_bbox_celeba.txt'), delim_whitespace=True, header=1,
                           index_col=0)
        landmark = pd.read_csv(os.path.join(self.raw_folder, 'list_landmarks_align_celeba.txt'),
                               delim_whitespace=True, header=1)
        split = pd.read_csv(os.path.join(self.raw_folder, 'list_eval_partition.txt'), delim_whitespace=True,
                            names=['split'], index_col=0)
        img = split.index.to_series()
        img = img.apply(lambda x: os.path.join(self.raw_folder, 'img_align_celeba', x))
        train_mask, test_mask = split['split'] <= 1, split['split'] == 2
        train_img, test_img = img[train_mask].tolist(), img[test_mask].tolist()
        train_attr, test_attr = (attr[train_mask].values + 1) // 2, (attr[test_mask].values + 1) // 2
        train_identity, test_identity = identity['identity'][train_mask].values, identity['identity'][test_mask].values
        train_bbox, test_bbox = bbox[train_mask].values, bbox[test_mask].values
        train_landmark, test_landmark = landmark[train_mask].values, landmark[test_mask].values
        train_target = {'attr': train_attr, 'identity': train_identity, 'bbox': train_bbox, 'landmark': train_landmark}
        test_target = {'attr': test_attr, 'identity': test_identity, 'bbox': test_bbox, 'landmark': test_landmark}
        classes_to_labels = {'attr': anytree.Node('U', index=[], flat_index=0),
                             'identity': anytree.Node('U', index=[], flat_index=0)}
        attr = attr.columns.tolist()
        identity = np.sort(identity['identity'].unique()).astype(str).tolist()
        for a in attr:
            make_tree(classes_to_labels['attr'], a)
        for i in identity:
            make_tree(classes_to_labels['identity'], i)
        classes_size = {'attr': make_flat_index(classes_to_labels['attr']),
                        'identity': make_flat_index(classes_to_labels['identity'])}
        return (train_img, train_target), (test_img, test_target), (classes_to_labels, classes_size)