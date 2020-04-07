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

data_name = 'Omniglot'
model_path = './output/model'
result_path = './output/result'
fig_path = './output/fig'
sub_path = 'test'
metric = 'test/InceptionScore'
num_Experiments = 10
control_exp = [str(x) for x in list(range(num_Experiments))]
control_mode_size = ['1', '0']
control_sharing_rate = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
if data_name == 'Omniglot':
    axis_exp = np.arange(num_Experiments)
    axis_mode_size = np.array([1, 32460])
else:
    raise ValueError('Not valid data name')


def main():
    result_control = {
        # 'CVAE': {
        #     'control_names': [control_exp, [data_name], ['label'], ['cvae'], control_mode_size]},
        # 'DCCVAE': {
        #     'control_names': [control_exp, [data_name], ['label'], ['dccvae'], control_mode_size]},
        # 'CGAN': {
        #     'control_names': [control_exp, [data_name], ['label'], ['cgan'], control_mode_size]},
        # 'DCCGAN': {
        #     'control_names': [control_exp, [data_name], ['label'], ['dccgan'], control_mode_size]},
        'MCVAE': {
            'control_names': [control_exp, [data_name], ['label'], ['mcvae'], control_mode_size,
            control_sharing_rate]},
        'DCMCVAE': {
            'control_names': [control_exp, [data_name], ['label'], ['dcmcvae'], control_mode_size,
                              control_sharing_rate]},
        # 'MCGAN': {
        #     'control_names': [[str(x) for x in list(range(num_Experiments))], [data_name], ['label'], ['mcgan'],
        #                       control_mode_size, control_sharing_rate]},
        # 'DCMCGAN': {
        #     'control_names': [[str(x) for x in list(range(num_Experiments))], [data_name], ['label'], ['dcmcgan'],
        #                       control_mode_size, control_sharing_rate]},
    }
    result = {}
    for result_name, info in result_control.items():
        result[result_name] = extract_result(result_name, info)
        print(result_name)
        print(result[result_name][:, :, 4])
    exit()
    show_result(result)
    # show_img()
    return


def extract_result(result_name, info):
    control_names = info['control_names']
    control_names_product = list(itertools.product(*control_names))
    if result_name in ['CVAE', 'DCCVAE', 'CGAN', 'DCCGAN']:
        extracted = np.zeros((len(control_exp), len(control_mode_size)))
        for i in range(len(control_names_product)):
            control_name = list(control_names_product[i])
            model_tag = '_'.join(control_name)
            path = '{}/{}/{}.pt'.format(result_path, sub_path, model_tag)
            if os.path.exists(path):
                result = load(path)
                logger = result['logger']
                exp_idx = control_exp.index(control_names_product[i][0])
                mode_size_idx = control_mode_size.index(control_names_product[i][4])
                extracted[exp_idx, mode_size_idx] = logger.mean[metric]
            else:
                print('Not valid model path {}'.format(path))
        return extracted
    else:
        extracted = np.zeros((len(control_exp), len(control_mode_size), len(control_sharing_rate)))
        for i in range(len(control_names_product)):
            control_name = list(control_names_product[i])
            model_tag = '_'.join(control_name)
            path = '{}/{}/{}.pt'.format(result_path, sub_path, model_tag)
            if os.path.exists(path):
                result = load(path)
                logger = result['logger']
                exp_idx = control_exp.index(control_names_product[i][0])
                mode_size_idx = control_mode_size.index(control_names_product[i][4])
                sharing_rate_idx = control_sharing_rate.index(control_names_product[i][5])
                extracted[exp_idx, mode_size_idx, sharing_rate_idx] = logger.mean[metric]
            else:
                print('Not valid model path {}'.format(path))
        return extracted


def show_result(result):
    fig_format = 'png'
    fontsize = 12
    linewidth = 3
    if_save = True
    figures = ['VAE', 'DCVAE', 'GAN', 'DCGAN']
    x_label = 'Data Size'
    y_label = 'Inception Score'
    colormap = plt.get_cmap('rainbow')
    colormap_indices = np.linspace(0, 1, len(control_sharing_rate)).tolist()
    for i in range(len(figures)):
        figure_name = figures[i]
        if figure_name in ['VAE', 'GAN']:
            conditional_figure_name = 'C' + figure_name
            mc_figure_name = 'MC' + figure_name
        elif figure_name in ['DCVAE', 'DCGAN']:
            conditional_figure_name = figure_name[:2] + 'C' + figure_name[2:]
            mc_figure_name = figure_name[:2] + 'MC' + figure_name[2:]
        fig = plt.figure()
        plt.plot(result[figure_name][0], result[figure_name][1], color='red', linestyle='-', label=figure_name,
                 linewidth=linewidth)
        plt.plot(result[conditional_figure_name][0], result[conditional_figure_name][1], color='orange', linestyle='--',
                 label=conditional_figure_name, linewidth=linewidth)
        for i in range(result[mc_figure_name][1].shape[0]):
            plt.plot(result[mc_figure_name][0][i], result[mc_figure_name][2][i], color=colormap(colormap_indices[i]),
                     linestyle='-', label='{}(r={})'.format(mc_figure_name, result[mc_figure_name][1][i, 0]),
                     linewidth=1)
        plt.legend(loc='lower right')
        plt.xscale('log')
        plt.rc('xtick', labelsize=fontsize)
        plt.rc('ytick', labelsize=fontsize)
        plt.xlabel(x_label, fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
        plt.yticks(np.arange(0, 11))
        plt.grid()
        if if_save:
            makedir_exist_ok('{}/{}'.format(fig_path, sub_path))
            fig.savefig('{}/{}/{}_{}.{}'.format(fig_path, sub_path, data_name, figure_name, fig_format), dpi=300,
                        bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
    plt.close()
    return


def show_img():
    fig_format = 'png'
    figures = ['VAE', 'DCVAE', 'GAN', 'DCGAN']
    save_per_mode = 5
    config.PARAM['classes_size'] = 10
    config.PARAM['data_name'] = data_name
    seed = 0
    load_tag = 'best'
    for i in range(len(figures)):
        figure_name = figures[i]
        if figure_name in ['VAE', 'GAN']:
            base_name = figure_name.lower()
            conditional_name = 'c' + base_name
            mc_name = 'mc' + base_name
        elif figure_name in ['DCVAE', 'DCGAN']:
            base_name = figure_name.lower()
            conditional_name = base_name[:2] + 'c' + base_name[2:]
            mc_name = base_name[:2] + 'mc' + base_name[2:]
        base_img = []
        conditional_img = []
        mc_img = [[] for _ in range(len(control_sharing_rate))]
        for j in range(len(control_mode_size)):
            config.PARAM['control'] = {'mode_data_size': control_mode_size[j]}
            control_name_list = []
            for k in config.PARAM['control']:
                control_name_list.append(config.PARAM['control'][k])
            config.PARAM['control_name'] = '_'.join(control_name_list)
            config.PARAM['model_name'] = base_name
            model_tag_list = [str(seed), config.PARAM['data_name'], config.PARAM['subset'],
                              config.PARAM['model_name'],
                              config.PARAM['control_name']]
            config.PARAM['model_tag'] = '_'.join(filter(None, model_tag_list))
            process_control_name()
            model = eval('models.{}().to(config.PARAM["device"])'.format(config.PARAM['model_name']))
            _, model, _, _, _ = resume(model, config.PARAM['model_tag'], load_tag=load_tag, verbose=False)
            base_img_j = model.generate(save_per_mode * config.PARAM['classes_size'])
            base_img_j = base_img_j.permute(1, 2, 3, 0).view(1, -1, base_img_j.size(0))
            base_img_j = F.fold(base_img_j, (32 * save_per_mode, 32 * config.PARAM['classes_size']), kernel_size=32,
                                stride=32)
            base_img.append(base_img_j)
            config.PARAM['model_name'] = conditional_name
            model_tag_list = [str(seed), config.PARAM['data_name'], config.PARAM['subset'],
                              config.PARAM['model_name'],
                              config.PARAM['control_name']]
            config.PARAM['model_tag'] = '_'.join(filter(None, model_tag_list))
            process_control_name()
            model = eval('models.{}().to(config.PARAM["device"])'.format(config.PARAM['model_name']))
            _, model, _, _, _ = resume(model, config.PARAM['model_tag'], load_tag=load_tag, verbose=False)
            conditional_img_j = model.generate(
                torch.arange(config.PARAM['classes_size']).to(config.PARAM['device']).repeat(save_per_mode))
            conditional_img_j = conditional_img_j.permute(1, 2, 3, 0).view(1, -1, conditional_img_j.size(0))
            conditional_img_j = F.fold(conditional_img_j, (32 * save_per_mode, 32 * config.PARAM['classes_size']),
                                       kernel_size=32,
                                       stride=32)
            conditional_img.append(conditional_img_j)
            for m in range(len(control_sharing_rate)):
                config.PARAM['control'] = {'mode_data_size': control_mode_size[j],
                                           'sharing_rate': control_sharing_rate[m]}
                control_name_list = []
                for k in config.PARAM['control']:
                    control_name_list.append(config.PARAM['control'][k])
                config.PARAM['control_name'] = '_'.join(control_name_list)
                config.PARAM['model_name'] = mc_name
                model_tag_list = [str(seed), config.PARAM['data_name'], config.PARAM['subset'],
                                  config.PARAM['model_name'],
                                  config.PARAM['control_name']]
                config.PARAM['model_tag'] = '_'.join(filter(None, model_tag_list))
                process_control_name()
                model = eval('models.{}().to(config.PARAM["device"])'.format(config.PARAM['model_name']))
                _, model, _, _, _ = resume(model, config.PARAM['model_tag'], load_tag=load_tag, verbose=False)
                mc_img_j_m = model.generate(
                    torch.arange(config.PARAM['classes_size']).to(config.PARAM['device']).repeat(save_per_mode))
                mc_img_j_m = mc_img_j_m.permute(1, 2, 3, 0).view(1, -1, mc_img_j_m.size(0))
                mc_img_j_m = F.fold(mc_img_j_m, (32 * save_per_mode, 32 * config.PARAM['classes_size']), kernel_size=32,
                                    stride=32)
                mc_img[m].append(mc_img_j_m)
        base_img = torch.cat(base_img, dim=0)
        conditional_img = torch.cat(conditional_img, dim=0)
        for m in range(len(mc_img)):
            mc_img[m] = torch.cat(mc_img[m], dim=0)
        img = torch.cat([base_img, conditional_img, *mc_img], dim=0)
        save_img(img, '{}/{}/{}_{}_generated.{}'.format(fig_path, sub_path, data_name, figure_name, fig_format),
                 nrow=len(control_mode_size), padding=2, pad_value=1)
    return


if __name__ == '__main__':
    main()