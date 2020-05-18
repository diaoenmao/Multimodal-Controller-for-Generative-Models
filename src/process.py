import config

config.init()
import os
import itertools
import matplotlib.pyplot as plt
import models
import torch
import torch.nn.functional as F
import numpy as np
from utils import save, load, to_device, process_control_name, process_dataset, resume, collate, save_img, \
    makedir_exist_ok
from torchvision.utils import make_grid

data_name = ['CIFAR10', 'Omniglot']
model_path = './output/model'
result_path = './output/result'
num_Experiments = 12
control_exp = [str(x) for x in list(range(num_Experiments))]


def main():
    processed_result = {}
    for i in range(len(data_name)):
        result_control = {
            'CVAE': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['cvae']], 'metric': 'test/MSE'},
            'MCVAE': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['mcvae'], ['0.5']], 'metric': 'test/MSE'},
            'VQVAE': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['vqvae']], 'metric': 'test/MSE'},
            'CPixelCNN': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['cpixelcnn']], 'metric': 'test/NLL'},
            'MCPixelCNN': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['mcpixelcnn'], ['0.5']],
                'metric': 'test/NLL'},
            'CGAN': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['cgan']], 'metric': None},
            'MCGAN': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['mcgan'], ['0.5']], 'metric': None},
            'CGLOW': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['cglow']], 'metric': 'test/Loss'},
            'MCGLOW': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['mcglow'], ['0.5']], 'metric': 'test/Loss'},
        }
        result = {}
        for result_name, info in result_control.items():
            result[result_name] = extract_result(info)
        processed_result[data_name[i]] = result
    print(processed_result)
    save(processed_result, '{}/processed_result.pt'.format(result_path))
    return


def extract_result(info):
    control_names = info['control_names']
    metric = info['metric']
    if metric is None:
        return
    control_names_product = list(itertools.product(*control_names))
    extracted = np.zeros(len(control_exp))
    for i in range(len(control_names_product)):
        control_name = list(control_names_product[i])
        model_tag = '_'.join(control_name)
        result_path_i = '{}/{}.pt'.format(result_path, model_tag)
        if os.path.exists(result_path_i):
            result = load(result_path_i)
            logger = result['logger']
            exp_idx = control_exp.index(control_names_product[i][0])
            extracted[exp_idx] = logger.mean[metric]
        else:
            print('Not valid result path {}'.format(result_path_i))
    result = {'mean': np.mean(extracted), 'stderr': np.std(extracted) / np.sqrt(num_Experiments)}
    return result


if __name__ == '__main__':
    main()