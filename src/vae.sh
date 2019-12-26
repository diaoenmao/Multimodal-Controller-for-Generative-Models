#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_vae.py --data_name MNIST --model_name vae --init_seed 0 --num_epochs 200 --control_name none_relu_1000_200_2_1_1&
CUDA_VISIBLE_DEVICES="1" python train_vae.py --data_name MNIST --model_name vae --init_seed 0 --num_epochs 200 --control_name none_relu_1000_200_2_10_1&
CUDA_VISIBLE_DEVICES="2" python train_vae.py --data_name MNIST --model_name vae --init_seed 0 --num_epochs 200 --control_name none_relu_1000_200_2_100_1&
CUDA_VISIBLE_DEVICES="3" python train_vae.py --data_name MNIST --model_name vae --init_seed 0 --num_epochs 200 --control_name none_relu_1000_200_2_500_1&
CUDA_VISIBLE_DEVICES="0" python train_vae.py --data_name MNIST --model_name vae --init_seed 0 --num_epochs 200 --control_name none_relu_1000_200_2_1000_1&
CUDA_VISIBLE_DEVICES="1" python train_vae.py --data_name MNIST --model_name vae --init_seed 0 --num_epochs 200 --control_name none_relu_1000_200_2_10000_1&
CUDA_VISIBLE_DEVICES="2" python train_vae.py --data_name MNIST --model_name vae --init_seed 0 --num_epochs 200 --control_name none_relu_1000_200_2_0_1&
CUDA_VISIBLE_DEVICES="0" python train_vae.py --data_name Omniglot --model_name vae --init_seed 0 --num_epochs 200 --control_name none_relu_1000_200_2_1_1&
CUDA_VISIBLE_DEVICES="1" python train_vae.py --data_name Omniglot --model_name vae --init_seed 0 --num_epochs 200 --control_name none_relu_1000_200_2_5_1&
CUDA_VISIBLE_DEVICES="2" python train_vae.py --data_name Omniglot --model_name vae --init_seed 0 --num_epochs 200 --control_name none_relu_1000_200_2_0_1&
