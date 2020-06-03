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
from metrics import Metric
from utils import save, to_device, process_control_name, process_dataset, resume, collate, save_img
from logger import Logger

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
config.PARAM['metric_names'] = {'test': ['DBI']}
config.PARAM['raw'] = False


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
    data_loader = make_data_loader(dataset)['train']
    if config.PARAM['raw']:
        metric = Metric()
        img, label = [], []
        for i, input in enumerate(data_loader):
            input = collate(input)
            img.append(input['img'])
            label.append(input['label'])
        img = torch.cat(img, dim=0)
        label = torch.cat(label, dim=0)
        output = {'img': img, 'label': label}
        evaluation = metric.evaluate(config.PARAM['metric_names']['test'], None, output)
        dbi_result = evaluation['DBI']
        print('Davies-Bouldin Index ({}): {}'.format(config.PARAM['data_name'], dbi_result))
        save(dbi_result, './output/result/dbi_{}.npy'.format(config.PARAM['data_name']), mode='numpy')
    else:
        created = np.load('./output/npy/created_{}.npy'.format(config.PARAM['model_tag']), allow_pickle=True)
        test(created)
    return


def test(created):
    with torch.no_grad():
        metric = Metric()
        created = torch.tensor(created / 255 * 2 - 1)
        valid_mask = torch.sum(torch.isnan(created), dim=(1, 2, 3)) == 0
        created = created[valid_mask]
        label = torch.arange(config.PARAM['classes_size'])
        label = label.repeat(config.PARAM['generate_per_mode'])
        label = label[valid_mask]
        output = {'img': created, 'label': label}
        evaluation = metric.evaluate(config.PARAM['metric_names']['test'], None, output)
    dbi_result = evaluation['DBI']
    print('Davies-Bouldin Index ({}): {}'.format(config.PARAM['model_tag'], dbi_result))
    save(dbi_result, './output/result/dbi_{}.npy'.format(config.PARAM['model_tag']), mode='numpy')
    return evaluation


if __name__ == "__main__":
    main()