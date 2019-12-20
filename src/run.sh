#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_vae.py --model_name \'fae\' --init_seed 0 --control_name \'32_32_2_0_1_0_0\' &
CUDA_VISIBLE_DEVICES="1" python train_vae.py --model_name \'fae\' --init_seed 0 --control_name \'32_32_2_0_1_0_1\' &
CUDA_VISIBLE_DEVICES="2" python train_vae.py --model_name \'fae\' --init_seed 0 --control_name \'32_32_2_0_10_0_0\' &
CUDA_VISIBLE_DEVICES="3" python train_vae.py --model_name \'fae\' --init_seed 0 --control_name \'32_32_2_0_10_0_1\' &
CUDA_VISIBLE_DEVICES="0" python train_vae.py --model_name \'fae\' --init_seed 0 --control_name \'32_32_2_0_50_0_0\' &
CUDA_VISIBLE_DEVICES="1" python train_vae.py --model_name \'fae\' --init_seed 0 --control_name \'32_32_2_0_50_0_1\' &
CUDA_VISIBLE_DEVICES="2" python train_vae.py --model_name \'fae\' --init_seed 0 --control_name \'32_32_2_0_100_0_0\' &
CUDA_VISIBLE_DEVICES="3" python train_vae.py --model_name \'fae\' --init_seed 0 --control_name \'32_32_2_0_100_0_1\' &
CUDA_VISIBLE_DEVICES="0" python train_vae.py --model_name \'fae\' --init_seed 0 --control_name \'32_32_2_0_500_0_0\' &
CUDA_VISIBLE_DEVICES="1" python train_vae.py --model_name \'fae\' --init_seed 0 --control_name \'32_32_2_0_500_0_1\' &
