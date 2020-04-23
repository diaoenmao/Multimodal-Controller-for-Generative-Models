import config

config.init()
import argparse
import datetime
import os
import torch
import torch.backends.cudnn as cudnn
import models
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control_name, process_dataset, resume, collate, save_img
from logger import Logger


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
    config.PARAM['model_tag'] = 'TF_{}'.format(config.PARAM['data_name'])
    print('Experiment: {}'.format(config.PARAM['model_tag']))
    runExperiment()
    return


def runExperiment():
    dataset = fetch_dataset(config.PARAM['data_name'], config.PARAM['subset'])
    process_dataset(dataset['train'])
    data_loader = make_data_loader(dataset)
    test(data_loader['train'])
    return


def test(data_loader):
    with torch.no_grad():
        generated = []
        for i, input in enumerate(data_loader):
            input = collate(input)
            generated.append(input['img'])
        generated = torch.cat(generated)
        generated = ((generated + 1) / 2 * 255).cpu().numpy()
        save(generated, './output/npy/{}.npy'.format(config.PARAM['model_tag']), mode='numpy')
    return


if __name__ == "__main__":
    main()