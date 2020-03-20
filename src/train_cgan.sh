#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name MNIST --model_name cgan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name MNIST --model_name cgan --init_seed 0 --num_epochs 200 --control_name 10&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name MNIST --model_name cgan --init_seed 0 --num_epochs 200 --control_name 100&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name MNIST --model_name cgan --init_seed 0 --num_epochs 200 --control_name 500&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name MNIST --model_name cgan --init_seed 0 --num_epochs 200 --control_name 1000&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name MNIST --model_name cgan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name MNIST --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name MNIST --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 10&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name MNIST --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 100&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name MNIST --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 500&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name MNIST --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 1000&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name MNIST --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="0" python train_gan.py --data_name Omniglot --model_name cgan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="1" python train_gan.py --data_name Omniglot --model_name cgan --init_seed 0 --num_epochs 200 --control_name 0&
CUDA_VISIBLE_DEVICES="2" python train_gan.py --data_name Omniglot --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 1&
CUDA_VISIBLE_DEVICES="3" python train_gan.py --data_name Omniglot --model_name dcgan --init_seed 0 --num_epochs 200 --control_name 0&
