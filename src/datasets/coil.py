import anytree
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index, \
    has_file_allowed_extension, \
    IMG_EXTENSIONS


class COIL100(Dataset):
    data_name = 'COIL100'
    file = [('http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip', None)]

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
        filenames = os.listdir(os.path.join(self.raw_folder, 'coil-100'))
        train_img, train_label = [], []
        test_img, test_label = [], []
        classes = set()
        for filename in filenames:
            if has_file_allowed_extension(filename, IMG_EXTENSIONS):
                train_img.append(os.path.join(self.raw_folder, 'coil-100', filename))
                test_img.append(os.path.join(self.raw_folder, 'coil-100', filename))
                classes.add(filename.split('_')[0])
        classes = sorted(list(classes))
        classes_to_labels = {'label': anytree.Node('U', index=[])}
        for c in classes:
            make_tree(classes_to_labels['label'], [c])
        classes_size = {'label': make_flat_index(classes_to_labels['label'])}
        r = anytree.resolver.Resolver()
        for i in range(len(train_img)):
            train_img_i = train_img[i]
            train_class_i = os.path.basename(train_img_i).split('_')[0]
            node = r.get(classes_to_labels['label'], train_class_i)
            train_label.append(node.flat_index)
        for i in range(len(test_img)):
            test_img_i = test_img[i]
            test_class_i = os.path.basename(test_img_i).split('_')[0]
            node = r.get(classes_to_labels['label'], test_class_i)
            test_label.append(node.flat_index)
        train_target = {'label': train_label}
        test_target = {'label': test_label}
        return (train_img, train_target), (test_img, test_target), (classes_to_labels, classes_size)