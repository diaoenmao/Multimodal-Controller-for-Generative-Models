import argparse
import datetime
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, resume, collate, save_img
from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])


def main():
    process_control()
    print('Experiment: {}'.format(cfg['data_name']))
    runExperiment()
    return


def runExperiment():
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
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
        generated = (generated + 1) / 2 * 255
        save(generated.numpy(), './output/npy/generated_0_{}.npy'.format(cfg['data_name']), mode='numpy')
    return


if __name__ == "__main__":
    main()