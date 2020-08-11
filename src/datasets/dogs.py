import anytree
import numpy as np
import os
import shutil
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index


class Dogs():
    data_name = 'Dogs'
    file = [('http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar', None),
            ('http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar', None),
            ('http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar', None),
            ('http://vision.stanford.edu/aditya86/ImageNetDogs/README.txt', None)]

    def __init__(self, root, split, subset, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.subset = subset
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.img, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))
        self.target = self.target[self.subset]
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.classes_size = load(os.path.join(self.processed_folder, 'meta.pt'))
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
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nSubset: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.subset, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        train_img = scipy.io.loadmat(os.path.join(self.raw_folder, 'train_list.mat'))['file_list']
        train_label = scipy.io.loadmat(os.path.join(self.raw_folder, 'train_list.mat'))['labels']
        test_img = scipy.io.loadmat(os.path.join(self.raw_folder, 'test_list.mat'))['file_list']
        test_label = scipy.io.loadmat(os.path.join(self.raw_folder, 'test_list.mat'))['labels']
        classes = list(sorted(set([train_img[i][0].item().split('/')[0] for i in range(len(train_img))])))
        train_img = [os.path.join(self.raw_folder, 'Images', train_img[i][0].item()) for i in range(len(train_img))]
        train_label = [train_label[i].item() for i in range(len(train_label))]
        test_img = [os.path.join(self.raw_folder, 'Images', test_img[i][0].item()) for i in range(len(test_img))]
        test_label = [test_label[i].item() for i in range(len(test_label))]
        ## No test
        train_img = train_img + test_img
        train_label = np.array(train_label + test_label, dtype=np.int64) - 1
        test_img = []
        test_label = np.array([], dtype=np.int64) - 1
        train_target, test_target = {'label': train_label}, {'label': test_label}
        classes_to_labels = {'label': anytree.Node('U', index=[])}
        for c in classes:
            make_tree(classes_to_labels['label'], [c])
        classes_size = {'label': make_flat_index(classes_to_labels['label'])}
        return (train_img, train_target), (test_img, test_target), (classes_to_labels, classes_size)