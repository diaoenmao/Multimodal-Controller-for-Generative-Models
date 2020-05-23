import config

config.init()
import argparse
import datetime
import os
import torch
import torch.backends.cudnn as cudnn
import models
import numpy as np
from data import fetch_dataset, make_data_loader
from utils import save, to_device, process_control_name, process_dataset, resume, collate, save_img

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Config')
for k in config.PARAM:
    exec('parser.add_argument(\'--{0}\',default=config.PARAM[\'{0}\'], type=type(config.PARAM[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in config.PARAM:
    config.PARAM[k] = args[k]
if args['control_name']:
    config.PARAM['control_name'] = args['control_name']
    if config.PARAM['control_name'] != 'None':
        control_list = list(config.PARAM['control'].keys())
        control_name_list = args['control_name'].split('_')
        for i in range(len(control_name_list)):
            config.PARAM['control'][control_list[i]] = control_name_list[i]
    else:
        config.PARAM['control'] = {}
else:
    if config.PARAM['control'] == 'None':
        config.PARAM['control'] = {}
control_name_list = []
for k in config.PARAM['control']:
    control_name_list.append(config.PARAM['control'][k])
config.PARAM['control_name'] = '_'.join(control_name_list)


def main():
    process_control_name()
    seeds = list(range(config.PARAM['init_seed'], config.PARAM['init_seed'] + config.PARAM['num_experiments']))
    for i in range(config.PARAM['num_experiments']):
        model_tag_list = [str(seeds[i]), config.PARAM['data_name'], config.PARAM['subset'], config.PARAM['model_name'],
                          config.PARAM['control_name']]
        model_tag_list = [x for x in model_tag_list if x]
        config.PARAM['model_tag'] = '_'.join(filter(None, model_tag_list))
        print('Experiment: {}'.format(config.PARAM['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(config.PARAM['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(config.PARAM['data_name'], config.PARAM['subset'])
    process_dataset(dataset['train'])
    model = eval('models.{}().to(config.PARAM["device"])'.format(config.PARAM['model_name']))
    _, model, _, _, _ = resume(model, config.PARAM['model_tag'], load_tag='best')
    transit(model)
    return


def transit(model, ae=None):
    with torch.no_grad():
        models.utils.create(model)
        model = model.to(config.PARAM['device'])
        model.train(False)
        if config.PARAM['data_name'] in ['Omniglot']:
            max_save_num_mode = [10, 100]
        else:
            max_save_num_mode = [100]
        root = 0
        save_num_step = 10
        alphas = np.linspace(0, 1, save_num_step + 1)
        for i in range(len(max_save_num_mode)):
            save_num_mode = min(max_save_num_mode[i], config.PARAM['classes_size'])
            C = torch.arange(save_num_mode).to(config.PARAM['device'])
            x = torch.randn([C.size(0), config.PARAM['latent_size']]).to(config.PARAM['device'])
            transited = []
            for j in range(len(alphas)):
                models.utils.transit(model, root, alphas[j])
                model = model.to(config.PARAM['device'])
                transited_i = model.generate(C, x)
                transited.append(transited_i.cpu())
            transited = torch.stack(transited, dim=0)
            transited = transited.view(-1, *transited.size()[2:])
            transited = (transited + 1) / 2
            save_img(transited, './output/img/transited_{}_{}.png'.format(config.PARAM['model_tag'], save_num_mode),
                     nrow=save_num_mode)
    return


if __name__ == "__main__":
    main()