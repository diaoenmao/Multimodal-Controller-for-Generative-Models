#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name MNIST --model_name gan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name MNIST --model_name gan --init_seed 0 --num_epochs 200 --control_name 10&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name MNIST --model_name gan --init_seed 0 --num_epochs 200 --control_name 100&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name MNIST --model_name gan --init_seed 0 --num_epochs 200 --control_name 500&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name MNIST --model_name gan --init_seed 0 --num_epochs 200 --control_name 1000&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name MNIST --model_name gan --init_seed 0 --num_epochs 200 --control_name 10000&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name MNIST --model_name gan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name MNIST --model_name cgan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name MNIST --model_name cgan --init_seed 0 --num_epochs 200 --control_name 10&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name MNIST --model_name cgan --init_seed 0 --num_epochs 200 --control_name 100&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name MNIST --model_name cgan --init_seed 0 --num_epochs 200 --control_name 500&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name MNIST --model_name cgan --init_seed 0 --num_epochs 200 --control_name 1000&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name MNIST --model_name cgan --init_seed 0 --num_epochs 200 --control_name 10000&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name MNIST --model_name cgan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name MNIST --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name MNIST --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 10&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name MNIST --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 100&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name MNIST --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 500&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name MNIST --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 1000&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name MNIST --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 10000&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name MNIST --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name MNIST --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name MNIST --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 10&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name MNIST --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 100&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name MNIST --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 500&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name MNIST --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 1000&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name MNIST --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 10000&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name MNIST --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name Omniglot --model_name gan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name Omniglot --model_name gan --init_seed 0 --num_epochs 200 --control_name 5&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name Omniglot --model_name gan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name Omniglot --model_name cgan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name Omniglot --model_name cgan --init_seed 0 --num_epochs 200 --control_name 5&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name Omniglot --model_name cgan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name Omniglot --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name Omniglot --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 5&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name Omniglot --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name Omniglot --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name Omniglot --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 5&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name Omniglot --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name CUB200 --model_name gan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name CUB200 --model_name gan --init_seed 0 --num_epochs 200 --control_name 5&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name CUB200 --model_name gan --init_seed 0 --num_epochs 200 --control_name 10&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name CUB200 --model_name gan --init_seed 0 --num_epochs 200 --control_name 20&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name CUB200 --model_name gan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name CUB200 --model_name cgan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name CUB200 --model_name cgan --init_seed 0 --num_epochs 200 --control_name 5&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name CUB200 --model_name cgan --init_seed 0 --num_epochs 200 --control_name 10&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name CUB200 --model_name cgan --init_seed 0 --num_epochs 200 --control_name 20&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name CUB200 --model_name cgan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name CUB200 --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name CUB200 --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 5&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name CUB200 --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 10&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name CUB200 --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 20&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name CUB200 --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name CUB200 --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name CUB200 --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 5&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name CUB200 --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 10&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name CUB200 --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 20&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name CUB200 --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name CelebA --model_name gan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name CelebA --model_name cgan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name CelebA --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name CelebA --model_name dccgan --init_seed 0 --num_epochs 200 --control_name 0&
