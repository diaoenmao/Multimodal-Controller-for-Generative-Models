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
                'control_names': [['1'], [data_name[i]], ['label'], ['cglow']], 'metric': 'test/Loss'},
            'MCGLOW': {
                'control_names': [['1'], [data_name[i]], ['label'], ['mcglow'], ['0.5']], 'metric': 'test/Loss'},
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
    control_names_product = list(itertools.product(*control_names))
    extracted = {'base': np.zeros(len(control_exp)), 'is': np.zeros((len(control_exp), 2)),
                 'fid': np.zeros(len(control_exp))}
    for i in range(len(control_names_product)):
        control_name = list(control_names_product[i])
        model_tag = '_'.join(control_name)
        if metric is not None:
            base_result_path_i = '{}/{}.pt'.format(result_path, model_tag)
            if os.path.exists(base_result_path_i):
                result = load(base_result_path_i)
                exp_idx = control_exp.index(control_names_product[i][0])
                extracted['base'][exp_idx] = result['logger'].mean[metric]
            else:
                pass
        is_result_path_i = '{}/is_{}.npy'.format(result_path, model_tag)
        if os.path.exists(is_result_path_i):
            result = np.load(is_result_path_i)
            exp_idx = control_exp.index(control_names_product[i][0])
            extracted['is'][exp_idx] = result
        else:
            pass
        fid_result_path_i = '{}/is_{}.npy'.format(result_path, model_tag)
        if os.path.exists(fid_result_path_i):
            result = load(fid_result_path_i, mode='numpy')
            exp_idx = control_exp.index(control_names_product[i][0])
            extracted['is'][exp_idx] = result
        else:
            pass
    result = {
        'base': {'mean': np.mean(extracted['base']), 'stderr': np.std(extracted['base']) / np.sqrt(num_Experiments)},
        'is': {'mean': np.mean(extracted['is'], axis=0),
               'stderr': np.std(extracted['is'], axis=0) / np.sqrt(num_Experiments)},
        'fid': {'mean': np.mean(extracted['fid']), 'stderr': np.std(extracted['fid']) / np.sqrt(num_Experiments)}}
    return result


if __name__ == '__main__':
    main()