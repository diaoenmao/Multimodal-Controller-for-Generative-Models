from .mnist import MNIST, EMNIST, FashionMNIST
from .celeba import CelebA
from .cifar import CIFAR10, CIFAR100
from .cub200 import CUB200
from .omniglot import Omniglot
from .svhn import SVHN
from .imagenet import ImageNet
from .folder import ImageFolder
from .utils import *
from .transforms import *

__all__ = ('MNIST','EMNIST', 'FashionMNIST',
           'CelebA',
           'CIFAR10', 'CIFAR100',
           'CUB200',
           'Omniglot',
            'SVHN',
           'ImageNet',
           'ImageFolder')
