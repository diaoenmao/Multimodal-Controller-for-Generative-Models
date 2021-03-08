from .mnist import MNIST, EMNIST, FashionMNIST
from .celebahq import CelebAHQ
from .cifar import CIFAR10, CIFAR100
from .cars import Cars
from .coil import COIL100
from .cub import CUB200
from .dogs import Dogs
from .omniglot import Omniglot
from .imagenet import ImageNet, ImageNet32, ImageNet64
from .utils import *
from .transforms import *

__all__ = ('MNIST',
           'CIFAR10', 'CIFAR100',
           'COIL100',
           'Omniglot')