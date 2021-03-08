import anytree
import numpy as np
import os
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index


class Cars(Dataset):
    data_name = 'Cars'
    file = [('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', None),
            ('http://imagenet.stanford.edu/internal/car196/car_devkit.tgz', None)]

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
        cars_annos = scipy.io.loadmat(os.path.join(self.raw_folder, 'cars_annos.mat'))['annotations']
        classes = scipy.io.loadmat(os.path.join(self.raw_folder, 'cars_annos.mat'))['class_names']
        classes = [classes[0, i].item() for i in range(classes.shape[1])]
        train_img, train_label = [], []
        test_img, test_label = [], []
        for i in range(cars_annos.shape[1]):
            test = cars_annos[0,i][-1].item()
            if not test:
                train_img.append(os.path.join(self.raw_folder, cars_annos[0, i][0].item()))
                train_label.append(cars_annos[0, i][-2].item())
            else:
                test_img.append(os.path.join(self.raw_folder, cars_annos[0, i][0].item()))
                test_label.append(cars_annos[0, i][-2].item())
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