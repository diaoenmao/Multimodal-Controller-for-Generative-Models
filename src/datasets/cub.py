import anytree
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index


class CUB200(Dataset):
    data_name = 'CUB200'
    file = [('http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz',
             '97eceeb196236b17998738112f37df78')]

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
            raise RuntimeError('Dataset not found')
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nSubset: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.subset, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        if not os.path.exists(os.path.join(self.raw_folder, 'CUB_200_2011')):
            extract_file(os.path.join(self.raw_folder, 'CUB_200_2011.tgz'))
        images = open(os.path.join(self.raw_folder, 'CUB_200_2011', 'images.txt'), 'r')
        image_class_labels = open(os.path.join(self.raw_folder, 'CUB_200_2011', 'image_class_labels.txt'), 'r')
        train_test_split = open(os.path.join(self.raw_folder, 'CUB_200_2011', 'train_test_split.txt'), 'r')
        classes = open(os.path.join(self.raw_folder, 'CUB_200_2011', 'classes.txt'), 'r')
        images = [x.split()[1] for x in images.readlines()]
        image_class_labels = [int(x.split()[1]) - 1 for x in image_class_labels.readlines()]
        train_test_split = [int(x.split()[1]) for x in train_test_split.readlines()]
        classes = [x.split()[1] for x in classes.readlines()]
        train_img, train_label = [], []
        test_img, test_label = [], []
        for i in range(len(images)):
            if train_test_split[i] in [0, 1]:
                train_img.append(os.path.join(self.raw_folder, 'CUB_200_2011', 'images', images[i]))
                train_label.append(image_class_labels[i])
            else:
                test_img.append(os.path.join(self.raw_folder, 'CUB_200_2011', 'images', images[i]))
                test_label.append(image_class_labels[i])

        train_target, test_target = {'label': train_label}, {'label': test_label}
        classes_to_labels = {'label': anytree.Node('U', index=[])}
        for c in classes:
            make_tree(classes_to_labels['label'], [c])
        classes_size = {'label': make_flat_index(classes_to_labels['label'])}
        return (train_img, train_target), (test_img, test_target), (classes_to_labels, classes_size)