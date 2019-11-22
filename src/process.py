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
sub_path = '02'
y_metric = 'test/PSNR'


def main():
    result_control = {
        'Split by subsets': {
            'evaluation_names': ['Federated BottleNeck', 'Isolated'],
            'control_names': [['0'], [data_name], ['faes'], ['32'], ['8'], ['32'], ['0'], ['2'],
                              ['1', '10', '50', '100', '500'], ['0'], ['0', '1']]},
        # 'Split by labels': {
        #     'evaluation_names': ['Federated BottleNeck', 'Isolated'],
        #     'control_names': [['0'], [data_name], ['faes'], ['32'], ['8'], ['32'], ['0'], ['2'],
        #                       ['1', '10', '50', '100', '500'], ['1'], ['0', '1']]}
    }
    for result_name, info in result_control.items():
        result = extract_result(info['control_names'], info)
        result = process_result(result)
        show_result(result, result_name)
    return


def extract_result(control_names, info):
    control_names_product = list(itertools.product(*control_names))
    x, y = {k: [] for k in info['evaluation_names']}, {k: [] for k in info['evaluation_names']}
    for i in range(len(control_names_product)):
        control_name = list(control_names_product[i])
        model_tag = '_'.join(control_name)
        path = '{}/{}/{}.pt'.format(result_path, sub_path, model_tag)
        if os.path.exists(path):
            result = load(path)
            logger = result['logger']
            if int(control_name[10]) == 0:
                x['Federated BottleNeck'].append(int(control_names_product[i][8]))
                y['Federated BottleNeck'].append(logger.mean[y_metric])
            else:
                x['Isolated'].append(int(control_names_product[i][8]))
                y['Isolated'].append(logger.mean[y_metric])
        else:
            print('Not valid model path {}'.format(path))
    return x, y


def process_result(result):
    x, y = result
    new_y = {k: {'mean': [], 'stderr': []} for k in x}
    for evaluation_name in x:
        x[evaluation_name] = np.array(x[evaluation_name])
        for i in range(len(x[evaluation_name])):
            np_y = np.array(y[evaluation_name][i])
            new_y[evaluation_name]['mean'].append(np.mean(np_y))
            if len(np_y) == 1:
                new_y[evaluation_name]['stderr'].append(0)
            else:
                new_y[evaluation_name]['stderr'].append(sem(np_y))
        new_y[evaluation_name]['mean'] = np.array(new_y[evaluation_name]['mean'])
        new_y[evaluation_name]['stderr'] = np.array(new_y[evaluation_name]['stderr'])
    return x, new_y


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