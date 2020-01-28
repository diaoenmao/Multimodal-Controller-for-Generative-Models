import config

config.init()
import argparse
import datetime
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import scipy
from utils import save, load, makedir_exist_ok
from logger import Logger

data_name = 'FashionMNIST'
result_path = './output/result'
fig_path = './output/fig'
sub_path = 'test'
y_metric = 'test/InceptionScore'
control_mode_size = ['1', '10', '100', '500', '1000', '0']
control_sharing_rate = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']


def main():
    result_control = {
        'VAE': {
            'evaluation_names': ['VAE', 'CVAE', 'DCVAE', 'DCCVAE'],
            'control_names': [['0'], [data_name], ['label'], ['vae', 'cvae', 'dcvae', 'dccvae'],
                              control_mode_size]},
        'GAN': {
            'evaluation_names': ['GAN', 'CGAN', 'DCGAN', 'DCCGAN'],
            'control_names': [['0'], [data_name], ['label'], ['gan', 'cgan', 'dcgan', 'dccgan'],
                              control_mode_size]},
        'RMVAE': {
            'evaluation_names': ['RMVAE'],
            'control_names': [['0'], [data_name], ['label'], ['rmvae'],
                              control_mode_size,
                              control_sharing_rate]},
        'DCRMVAE': {
            'evaluation_names': ['DCRMVAE'],
            'control_names': [['0'], [data_name], ['label'], ['dcrmvae'],
                              control_mode_size,
                              control_sharing_rate]},
        'RMGAN': {
            'evaluation_names': ['RMGAN'],
            'control_names': [['0'], [data_name], ['label'], ['rmgan'],
                              control_mode_size,
                              control_sharing_rate]},
        'DCRMGAN': {
            'evaluation_names': ['DCRMGAN'],
            'control_names': [['0'], [data_name], ['label'], ['dcrmgan'],
                              control_mode_size,
                              control_sharing_rate]},
    }
    for result_name, info in result_control.items():
        result = extract_result(result_name, info)
        # print(result)
        print(result_name)
        print(result[-1][result_name])
        # show_result(result, result_name)
    return


def extract_result(result_name, info):
    control_names = info['control_names']
    control_names_product = list(itertools.product(*control_names))
    if result_name in ['VAE', 'GAN']:
        x = np.array([10, 100, 1000, 5000, 10000, 60000])
        x, y = {k: x for k in info['evaluation_names']}, {k: np.zeros(x.shape[0]) for k in info['evaluation_names']}
        for i in range(len(control_names_product)):
            control_name = list(control_names_product[i])
            model_tag = '_'.join(control_name)
            path = '{}/{}/{}.pt'.format(result_path, sub_path, model_tag)
            if os.path.exists(path):
                result = load(path)
                logger = result['logger']
                x_idx = control_mode_size.index(control_names_product[i][4])
                evaluation_names_i = control_name[3].upper()
                y[evaluation_names_i][x_idx] = logger.mean[y_metric]
            else:
                print('Not valid model path {}'.format(path))
        return x, y
    else:
        x, y = np.meshgrid(np.array([10, 100, 1000, 5000, 10000, 60000]),np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))
        z = np.zeros(x.shape)
        for i in range(len(control_names_product)):
            control_name = list(control_names_product[i])
            model_tag = '_'.join(control_name)
            path = '{}/{}/{}.pt'.format(result_path, sub_path, model_tag)
            if os.path.exists(path):
                result = load(path)
                logger = result['logger']
                x_idx = control_mode_size.index(control_names_product[i][4])
                y_idx = control_sharing_rate.index(control_names_product[i][5])
                z[y_idx, x_idx] = logger.mean[y_metric]
            else:
                print('Not valid model path {}'.format(path))
        x, y, z = {info['evaluation_names'][0]: x}, {info['evaluation_names'][0]: y}, {info['evaluation_names'][0]: z}
        return x, y, z


def show_result(result, result_name):
    x, y = result
    fig_format = 'png'
    color = {'Federated BottleNeck': 'red', 'Isolated': 'blue'}
    linestyle = {'Federated BottleNeck': '-', 'Isolated': '-'}
    num_stderr = 1.96
    fontsize = 20
    linewidth = 3
    x_label = 'number of devices'
    y_label = 'PSNR (db)'
    ifsave = True
    fig = plt.figure()
    plt.rc('xtick', labelsize=fontsize - 8)
    plt.rc('ytick', labelsize=fontsize - 8)
    plt.grid()
    for evaluation_name in x:
        plt.plot(x[evaluation_name], y[evaluation_name]['mean'], color=color[evaluation_name],
                 linestyle=linestyle[evaluation_name], label=evaluation_name, linewidth=linewidth)
        if num_stderr > 0:
            plt.fill_between(x[evaluation_name], y[evaluation_name]['mean'] + num_stderr * y[evaluation_name]['stderr'],
                             y[evaluation_name]['mean'] - num_stderr * y[evaluation_name]['stderr'],
                             color=color[evaluation_name], alpha=0.5, linewidth=1)
        plt.xlabel(x_label, fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
    plt.legend()
    if ifsave:
        makedir_exist_ok('{}/{}'.format(fig_path, sub_path))
        fig.savefig('{}/{}/{}_{}.{}'.format(fig_path, sub_path, data_name, result_name, fig_format), dpi=300,
                    bbox_inches='tight',
                    pad_inches=0)
    else:
        plt.show()
    plt.close()
    return


if __name__ == '__main__':
    main()