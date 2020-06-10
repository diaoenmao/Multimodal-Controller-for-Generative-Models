# Multimodal Controller for Generative Models
This is an implementation of Multimodal Controller for Generative Models
 
## Requirements
See requirements.txt

## Instruction
 - Global hyperparameters are configured in config.py
 - Model hyperparameters can be found at ./src/utils.py (process_control_name)
 - Training and testing hyperparameters can be found at training and testing files corresponding with each model
 - Transition and Creation methods can be found at ./src/models/utils.py

## Example
 - Train a MCGAN with CIFAR10
    ```ruby
    python train_gan.py --data_name CIFAR10 --model_name mcgan --control_name 0.5
    ```
 - Train a CGAN with Omniglot
    ```ruby
    python train_gan.py --data_name Omniglot --model_name cgan --control_name None
    ```
 - Train a VQ-VAE with Omniglot
    ```ruby
    python train_vqvae.py --data_name Omniglot --model_name vqvae --control_name None
    ```
 - Test a MCVAE with CIFAR10
    ```ruby
    python test_vae.py --data_name CIFAR10 --model_name mcvae --control_name 0.5
    ```
 - Generate/Transit/Create from MCGlow trained with Omniglot
    * 'save_npy' has to be True in config.py to test generations
    * 'save_img' has to be True in config.py to plot images
    * 'save_npy' has to be False and 'save_img' has to be True to plot images from different number of modes
    ```ruby
    python generate/transit/create.py --data_name Omniglot --model_name mcglow --control_name 0.5
    ```
 - Test generations from MCVAE (seed=0) trained with CIFAR10 for IS
    ```ruby
    python ./metrics_tf/inception_score_tf.py npy generated_0_CIFAR10_label_mcvae_0.5
    ```
 - Test generations from CPixlecnn (seed=0) trained with CIFAR10 for FID
    ```ruby
    python ./metrics_tf/fid_tf.py npy generated_0_CIFAR10_label_cpixelcnn
    ```
 - Test generations from MCGlow (seed=0) trained with CIFAR10 for IS and FID
    ```ruby
    python test_generated.py --init_seed 0 --data_name CIFAR10 --model_name mcglow --control_name 0.5
    ```
 - Test creations from MCGAN (seed=0) trained with Omniglot for DBI
    ```ruby
    python test_created.py --init_seed 0 --data_name Omniglot --model_name mcgan --control_name 0.5
    ```
 - Summarize model architecture of CGAN
    ```ruby
    python summarize.py --model_name cgan --control_name None
    ```