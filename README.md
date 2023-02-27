# Multimodal Controller for Generative Models
[CVMI 2022] This is an implementation of [Multimodal Controller for Generative Models](https://arxiv.org/abs/2002.02572)
- Multimodal Controlled Neural Networks
<img src="/assest/mc.png">


## Requirements
 - see requirements.txt

## Instruction

 - Global hyperparameters are configured in config.yml
 - Hyperparameters can be found at process_control() in utils.py 
 - MultimodalController can be found at ./src/modules/modules.py 
 - Creation methods can be found at ./src/models/utils.py

## Examples
 - Train a MCGAN with CIFAR10
    ```ruby
    python train_gan.py --data_name CIFAR10 --model_name mcgan --control_name 0.5
    ```
 - Train a CGAN with Omniglot
    ```ruby
    python train_gan.py --data_name Omniglot --model_name cgan --control_name None
    ```
 - Generate/Transit/Create from MCGlow trained with Omniglot
    * 'save_npy=True' in config.py to test generations
    * 'save_img=True' in config.py to plot images
    * 'save_npy=False' and 'save_img=True' to plot images from different number of modes
    ```ruby
    python generate/transit/create.py --data_name Omniglot --model_name mcglow --control_name 0.5
    ```
 - Test generations from MCVAE (seed=0) trained with CIFAR10 for IS
    ```ruby
    python ./metrics_tf/inception_score_tf.py npy generated_0_CIFAR10_label_mcvae_0.5
    ```
 - Test generations from CPixlecnn (seed=0) trained with COIL100 for FID
    ```ruby
    python ./metrics_tf/fid_tf.py npy generated_0_COIL100_label_cpixelcnn
    ```
 - Test generations from MCGlow (seed=0) trained with COIL100 for IS and FID
    ```ruby
    python test_generated.py --init_seed 0 --data_name COIL100 --model_name mcglow --control_name 0.5
    ```
 - Test creations from MCGAN (seed=0) trained with Omniglot for DBI
    ```ruby
    python test_created.py --init_seed 0 --data_name Omniglot --model_name mcgan --control_name 0.5
    ```
## Results
- a) MCGAN (b) CGAN trained with COIL100 dataset. Generations in each column are from one data modality.
![MNIST_interp_iid](/assest/Generation_GAN_COIL100.png)

- a) MCGAN (b) CGAN trained with Omniglot dataset. Generations in each column are from one data modality.
![MNIST_interp_iid](/assest/Generation_GAN_Omniglot.png)

- (a,b) MCGAN  and  (c,d)  CGAN  trained  with COIL100 dataset. Transitions in each column are created from interpolations from the first data modality to others. Uniform data modalities are created from resam-pling of pre-trained data modalities.
![MNIST_interp_iid](/assest/Creation_GAN_COIL100.png)

- (a,b) MCGAN  and  (c,d)  CGAN  trained  with COIL100 dataset. Transitions in each column are created from interpolations from the first data modality to others. Uniform data modalities are created from resam-pling of pre-trained data modalities.
![MNIST_interp_iid](/assest/Creation_GAN_Omniglot.png)

## Acknowledgement
*Enmao Diao  
Jie Ding  
Vahid Tarokh*
