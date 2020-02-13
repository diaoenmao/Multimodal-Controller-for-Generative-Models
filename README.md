# Multimodal Controller for Generative Models

This is an implementation of Multimodal Controller for Generative Models
 
## Requirements
See requirements.txt


## Instruction
 - Global hyperparameters are configured in config.py
 - (DC)VAE and (DC)GAN are trained (tested) with train_vae.py (test_vae.py) and train_gan.py (test_gan.py)
 - VAE and GAN have their own hyperparameters set in rain_vae.py (test_vae.py) and train_gan.py (test_gan.py)
 - Models are described in **./models/vae.py**, **./models/gan.py** and **./models/classifier.py**
 - Modules including **MultimodalController** are described in **./moduels/cell.py**
 - **./output/model/0_MNIST_label_classifier_0_best.pt** and **./output/model/0_FashionMNIST_label_classifier_0_best.pt** are the trained classifiers we use to compute Inception Score

## Example
 - Baseline VAE trained with 100 data points per mode
    * Change value of *'model_name'* in **config.py** to *'vae'* and value of *'control'* to *{'mode_data_size': '100'}*
 - Multimodal controlled DCGAN trained with full dataset and 0.5 sharing rate 
    * Change value of *'model_name'* in **config.py** to *'dcmcgan'* and value of *'control'* to *{'mode_data_size': '0', 'sharing_rate': '0.5'}*