from .mnist import MNIST, FashionMNIST
from .cifar import CIFAR10, CIFAR100
from .coil import COIL100
from .omniglot import Omniglot
from .utils import *

__all__ = ('MNIST',
           'CIFAR10', 'CIFAR100',
           'COIL100',
           'Omniglot')