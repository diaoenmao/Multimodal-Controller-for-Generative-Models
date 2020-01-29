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

data_name = 'MNIST'
result_path = './output/result'
fig_path = './output/fig'
sub_path = 'test'
y_metric = 'test/InceptionScore'
control_mode_size = ['1', '10', '100', '500', '1000', '0']
control_sharing_rate = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']


def main():
    result_control = {
        'VAE': {
            'control_names': [['0'], [data_name], ['label'], ['vae'],
                              control_mode_size]},
        'CVAE': {
            'control_names': [['0'], [data_name], ['label'], ['cvae'],
                              control_mode_size]},
        'DCVAE': {
            'control_names': [['0'], [data_name], ['label'], ['dcvae'],
                              control_mode_size]},
        'DCCVAE': {
            'control_names': [['0'], [data_name], ['label'], ['dccvae'],
                              control_mode_size]},
        'GAN': {
            'control_names': [['0'], [data_name], ['label'], ['gan'],
                              control_mode_size]},
        'CGAN': {
            'control_names': [['0'], [data_name], ['label'], ['cgan'],
                              control_mode_size]},
        'DCGAN': {
            'control_names': [['0'], [data_name], ['label'], ['dcgan'],
                              control_mode_size]},
        'DCCGAN': {
            'control_names': [['0'], [data_name], ['label'], ['dccgan'],
                              control_mode_size]},
        'RMVAE': {
            'control_names': [['0'], [data_name], ['label'], ['rmvae'],
                              control_mode_size,
                              control_sharing_rate]},
        'DCRMVAE': {
            'control_names': [['0'], [data_name], ['label'], ['dcrmvae'],
                              control_mode_size,
                              control_sharing_rate]},
        'RMGAN': {
            'control_names': [['0'], [data_name], ['label'], ['rmgan'],
                              control_mode_size,
                              control_sharing_rate]},
        'DCRMGAN': {
            'control_names': [['0'], [data_name], ['label'], ['dcrmgan'],
                              control_mode_size,
                              control_sharing_rate]},
    }
    result = {}
    for result_name, info in result_control.items():
        result[result_name] = extract_result(result_name, info)
        # print(result_name)
        # print(result[result_name])
    show_result(result)
    return


def extract_result(result_name, info):
    control_names = info['control_names']
    control_names_product = list(itertools.product(*control_names))
    if result_name in ['VAE', 'CVAE', 'DCVAE', 'DCCVAE', 'GAN', 'CGAN', 'DCGAN', 'DCCGAN']:
        x = np.array([10, 100, 1000, 5000, 10000, 60000])
        y = np.zeros(x.shape[0])
        for i in range(len(control_names_product)):
            control_name = list(control_names_product[i])
            model_tag = '_'.join(control_name)
            path = '{}/{}/{}.pt'.format(result_path, sub_path, model_tag)
            if os.path.exists(path):
                result = load(path)
                logger = result['logger']
                x_idx = control_mode_size.index(control_names_product[i][4])
                y[x_idx] = logger.mean[y_metric]
            else:
                print('Not valid model path {}'.format(path))
        return x, y
    else:
        x, y = np.meshgrid(np.array([10, 100, 1000, 5000, 10000, 60000]),
                           np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))
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
        return x, y, z


def show_result(result):
    fig_format = 'png'
    fontsize = 12
    linewidth = 3
    if_save = True
    figures = ['VAE', 'DCVAE', 'GAN', 'DCGAN']
    x_label = 'Data Size'
    y_label = 'Inception Score'
    colormap = plt.get_cmap('rainbow')
    colormap_indices = np.linspace(0,1,len(control_sharing_rate)).tolist()
    for i in range(len(figures)):
        figure_name = figures[i]
        if figure_name in ['VAE', 'GAN']:
            conditional_figure_name = 'C'+figure_name
            rm_figure_name = 'RM'+figure_name
        elif figure_name in ['DCVAE', 'DCGAN']:
            conditional_figure_name = figure_name[:2] + 'C'+ figure_name[2:]
            rm_figure_name = figure_name[:2] + 'RM' + figure_name[2:]
        fig = plt.figure()
        plt.plot(result[figure_name][0], result[figure_name][1], color='red', linestyle='-', label=figure_name, linewidth=linewidth)
        plt.plot(result[conditional_figure_name][0], result[conditional_figure_name][1], color='orange', linestyle='--', label=conditional_figure_name, linewidth=linewidth)
        for i in range(result[rm_figure_name][1].shape[0]):
            plt.plot(result[rm_figure_name][0][i], result[rm_figure_name][2][i], color=colormap(colormap_indices[i]), linestyle='-', label='{}(r={})'.format(rm_figure_name, result[rm_figure_name][1][i,0]),
                     linewidth=1)
        plt.legend(loc='lower right')
        plt.xscale('log')
        plt.rc('xtick', labelsize=fontsize)
        plt.rc('ytick', labelsize=fontsize)
        plt.xlabel(x_label, fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
        plt.grid()
        if if_save:
            makedir_exist_ok('{}/{}'.format(fig_path, sub_path))
            fig.savefig('{}/{}/{}_{}.{}'.format(fig_path, sub_path, data_name, figure_name, fig_format), dpi=300,
                        bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
    plt.close()
    return


if __name__ == '__main__':
    main()