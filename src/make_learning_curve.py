import os
import itertools
import json
import numpy as np
from utils import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
import matplotlib.cm as cm

model_path = './output/model'
vis_path = './output/vis/lc'
num_experiments = 1
exp = [str(x) for x in list(range(num_experiments))]
colors = {'c': 'b', 'mc': 'r'}
fig_format = 'pdf'
metrics = ['test/InceptionScore', 'test/FID']


def main():
    c_controls = [exp, ['CIFAR10', 'CIFAR100', 'COIL100', 'Omniglot'], ['label'], ['cgan']]
    c_controls = list(itertools.product(*c_controls))
    mc_controls = [exp, ['CIFAR10', 'CIFAR100', 'COIL100', 'Omniglot'], ['label'], ['mcgan'], ['0.5']]
    mc_controls = list(itertools.product(*mc_controls))
    controls = c_controls + mc_controls
    processed_result = process_result(controls)
    make_vis(processed_result)
    return


def process_result(controls):
    processed_result = {}
    for control in controls:
        model_tag = '_'.join(control)
        model_path_i = os.path.join(model_path, '{}_checkpoint.pt'.format(model_tag))
        if os.path.exists(model_path_i):
            model = load(model_path_i)
            extract_result(list(control), processed_result, model)
    summarize_result(processed_result)
    return processed_result


def extract_result(control, processed_result, model):
    if len(control) == 1:
        for metric in metrics:
            if metric not in processed_result:
                processed_result[metric] = {'hist': [None for _ in range(num_experiments)]}
            exp_idx = exp.index(control[0])
            processed_result[metric]['hist'][exp_idx] = np.array(model['logger'].history[metric])
    else:
        if control[1] not in processed_result:
            processed_result[control[1]] = {}
        extract_result([control[0]] + control[2:], processed_result[control[1]], model)
    return


def summarize_result(processed_result):
    if metrics[0] in processed_result:
        for metric in metrics:
            processed_result[metric]['hist'] = np.stack(processed_result[metric]['hist'], axis=0)
            processed_result[metric]['hist_mean'] = np.mean(processed_result[metric]['hist'], axis=0)
            processed_result[metric]['hist_std'] = np.std(processed_result[metric]['hist'], axis=0)
            processed_result[metric]['hist_max'] = np.max(processed_result[metric]['hist'], axis=0)
            processed_result[metric]['hist_min'] = np.min(processed_result[metric]['hist'], axis=0)
            processed_result[metric]['hist_min'] = np.min(processed_result[metric]['hist'], axis=0)
            processed_result[metric]['hist_argmax'] = np.argmax(processed_result[metric]['hist'], axis=0)
            processed_result[metric]['hist_argmin'] = np.argmin(processed_result[metric]['hist'], axis=0)
    else:
        for k, v in processed_result.items():
            summarize_result(v)
    return


def make_vis(processed_result):
    fig = {}
    vis_result(fig, [], processed_result)
    for k, v in fig.items():
        metric_name = k.split('_')[-1].split('/')
        save_fig_name = '_'.join(k.split('_')[:-1] + ['_'.join(metric_name)])
        fig[k] = plt.figure(k)
        plt.xlabel('iter', fontsize=12)
        plt.ylabel(metric_name[-1], fontsize=12)
        plt.grid()
        plt.legend()
        fig_path = '{}/{}.{}'.format(vis_path, save_fig_name, fig_format)
        makedir_exist_ok(vis_path)
        fig[k].savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(k)
    return


def vis_result(fig, controls, processed_result):
    if metrics[0] in processed_result:
        for metric in metrics:
            data_name = controls[0]
            label = controls[2]
            if 'mc' in label:
                color = colors['mc']
                model_name = label[2:]
            else:
                color = colors['c']
                model_name = label[1:]
            label = label.upper()
            y_mean = processed_result[metric]['hist_mean']
            x = np.arange(y_mean.shape[0])
            y_std = processed_result[metric]['hist_std']
            fig_name = '{}_{}_{}'.format(data_name, model_name, metric)
            fig[fig_name] = plt.figure(fig_name)
            plt.plot(x, y_mean, c=color, label=label)
    else:
        for k, v in processed_result.items():
            vis_result(fig, controls + [k], v)
    return


if __name__ == '__main__':
    main()