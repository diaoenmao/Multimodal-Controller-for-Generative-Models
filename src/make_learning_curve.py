import os
import itertools
import numpy as np
from config import cfg
from utils import load, makedir_exist_ok
import matplotlib.pyplot as plt

model_path = './output/model'
vis_path = './output/vis/lc'
num_experiments = 1
exp = [str(x) for x in list(range(num_experiments))]
data_names = ['CIFAR10', 'CIFAR100', 'COIL100', 'Omniglot']
colors = {'c': 'b', 'mc': 'r'}


def main():
    cgan_control = [exp, data_names, ['label'], ['cgan']]
    mcgan_control = [exp, data_names, ['label'], ['mcgan'], ['0.5']]
    controls_list = [cgan_control, mcgan_control]
    controls = []
    for i in range(len(controls_list)):
        controls.extend(list(itertools.product(*controls_list[i])))
    processed_result = process_result(controls)
    make_vis(processed_result)
    return


def process_result(controls):
    processed_result = {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), processed_result, model_tag)
    summarize_result(processed_result)
    return processed_result


def extract_result(control, processed_result, model_tag):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        model_path_i = os.path.join(model_path, '{}_checkpoint.pt'.format(model_tag))
        if os.path.exists(model_path_i):
            model = load(model_path_i)
            if 'FID' not in processed_result:
                processed_result['IS'] = {'hist': [None for _ in range(num_experiments)]}
            processed_result['IS']['hist'][exp_idx] = np.array(model['logger'].history['test/InceptionScore'])
            if 'FID' not in processed_result:
                processed_result['FID'] = {'hist': [None for _ in range(num_experiments)]}
            processed_result['FID']['hist'][exp_idx] = np.array(model['logger'].history['test/FID'])
    else:
        if control[1] not in processed_result:
            processed_result[control[1]] = {}
        extract_result([control[0]] + control[2:], processed_result[control[1]], model_tag)
    return


def summarize_result(processed_result):
    if 'hist' in processed_result:
        processed_result['hist'] = np.stack(processed_result['hist'], axis=0)
        processed_result['mean'] = np.mean(processed_result['hist'], axis=0)
        processed_result['std'] = np.std(processed_result['hist'], axis=0)
        processed_result['max'] = np.max(processed_result['hist'], axis=0)
        processed_result['min'] = np.min(processed_result['hist'], axis=0)
        processed_result['argmax'] = np.argmax(processed_result['hist'], axis=0)
        processed_result['argmin'] = np.argmin(processed_result['hist'], axis=0)
        processed_result['hist'] = processed_result['hist'].tolist()
    else:
        for k, v in processed_result.items():
            summarize_result(v)
    return


def make_vis(processed_result):
    fig = {}
    vis(fig, [], processed_result)
    for k, v in fig.items():
        metric_name = k.split('_')[-1].split('/')
        save_fig_name = '_'.join(k.split('_')[:-1] + ['_'.join(metric_name)])
        fig[k] = plt.figure(k)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel(metric_name[-1], fontsize=15)
        plt.grid()
        if metric_name[-1] == 'FID':
            plt.legend(loc='upper right', fontsize=15)
        else:
            plt.legend(loc='lower right', fontsize=15)
        fig_path = '{}/{}.{}'.format(vis_path, save_fig_name, cfg['save_format'])
        makedir_exist_ok(vis_path)
        fig[k].savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(k)
    return


def vis(fig, control, processed_result):
    if 'hist' in processed_result:
        data_name = control[0]
        model_name = control[2]
        metric = control[-1]
        if 'mc' in model_name:
            color = colors['mc']
            label = model_name[2:]
        else:
            color = colors['c']
            label = model_name[1:]
        model_name = model_name.upper()
        y_mean = processed_result['mean']
        x = np.arange(y_mean.shape[0])
        y_std = processed_result['std']
        fig_name = '{}_{}_{}'.format(data_name, label, metric)
        fig[fig_name] = plt.figure(fig_name)
        plt.plot(x, y_mean, c=color, label=model_name)
    else:
        for k, v in processed_result.items():
            vis(fig, control + [k], v)
    return


if __name__ == '__main__':
    main()