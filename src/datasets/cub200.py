import anytree
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index


class CUB200(Dataset):
    data_name = 'CUB200'
    file = [
        ('http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz',
         '97eceeb196236b17998738112f37df78')
    ]

    def __init__(self, root, split, subset, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.subset = subset
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.img, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))
        self.classes_to_labels, self.classes_size = load(os.path.join(self.processed_folder, 'meta.pt'))
        if self.subset == 'label':
            self.classes_to_labels, self.classes_size = self.classes_to_labels[self.subset], \
                                                        self.classes_size[self.subset]
            self.classes_counts = make_classes_counts(self.target[self.subset])

    def __getitem__(self, index):
        img, target = Image.open(self.img[index], mode='r').convert('RGB'), \
                      {s: torch.tensor(self.target[s][index]) for s in self.target}
        input = {'img': img, **target}
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
        img = pd.read_csv(os.path.join(self.raw_folder, 'CUB_200_2011', 'images.txt'), delim_whitespace=True,
                          names=['id', 'img'])
        label = pd.read_csv(os.path.join(self.raw_folder, 'CUB_200_2011', 'image_class_labels.txt'),
                            delim_whitespace=True, names=['id', 'label'])
        class_label = pd.read_csv(os.path.join(self.raw_folder, 'CUB_200_2011', 'classes.txt'), delim_whitespace=True,
                                  names=['label', 'class'])
        bbox = pd.read_csv(os.path.join(self.raw_folder, 'CUB_200_2011', 'bounding_boxes.txt'), delim_whitespace=True,
                           names=['id', 'x', 'y', 'width', 'height'])
        split = pd.read_csv(os.path.join(self.raw_folder, 'CUB_200_2011', 'train_test_split.txt'),
                            delim_whitespace=True, names=['id', 'split'])
        img['img'] = img['img'].apply(lambda x: os.path.join(self.raw_folder, 'CUB_200_2011', 'images', x))
        class_label['class'] = class_label['class'].apply(lambda x: x[4:])
        label = pd.merge(label, class_label, on='label')
        train_mask, test_mask = split['split'] == 1, split['split'] == 0
        train_img, test_img = img['img'][train_mask].values.tolist(), img['img'][test_mask].values.tolist()
        train_label, test_label = (label['label'][train_mask].values - 1), (label['label'][test_mask].values - 1)
        train_bbox, test_bbox = bbox[['x', 'y', 'width', 'height']][train_mask].values, \
                                bbox[['x', 'y', 'width', 'height']][test_mask].values
        train_target = {'label': train_label, 'bbox': train_bbox}
        test_target = {'label': test_label, 'bbox': test_bbox}
        classes_to_labels = {'label': anytree.Node('U', index=[])}
        for i in range(class_label['class'].values.shape[0]):
            make_tree(classes_to_labels['label'], [class_label['class'].values[i]])
        classes_size = {'label': make_flat_index(classes_to_labels['label'])}
        return (train_img, train_target), (test_img, test_target), (classes_to_labels, classes_size)