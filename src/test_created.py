import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
import numpy as np
from config import cfg
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, process_control, process_dataset, collate

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

cfg['metric_names'] = {'test': ['DBI']}


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    process_dataset(dataset['train'])
    if cfg['raw']:
        data_loader = make_data_loader(dataset)['train']
        metric = Metric()
        img, label = [], []
        for i, input in enumerate(data_loader):
            input = collate(input)
            img.append(input['img'])
            label.append(input['label'])
        img = torch.cat(img, dim=0)
        label = torch.cat(label, dim=0)
        output = {'img': img, 'label': label}
        evaluation = metric.evaluate(cfg['metric_names']['test'], None, output)
        dbi_result = evaluation['DBI']
        print('Davies-Bouldin Index ({}): {}'.format(cfg['data_name'], dbi_result))
        save(dbi_result, './output/result/dbi_{}.npy'.format(cfg['data_name']), mode='numpy')
    else:
        created = np.load('./output/npy/created_{}.npy'.format(cfg['model_tag']), allow_pickle=True)
        test(created)
    return


def test(created):
    with torch.no_grad():
        metric = Metric()
        created = torch.tensor(created / 255 * 2 - 1)
        valid_mask = torch.sum(torch.isnan(created), dim=(1, 2, 3)) == 0
        created = created[valid_mask]
        label = torch.arange(cfg['classes_size'])
        label = label.repeat(cfg['generate_per_mode'])
        label = label[valid_mask]
        output = {'img': created, 'label': label}
        evaluation = metric.evaluate(cfg['metric_names']['test'], None, output)
    dbi_result = evaluation['DBI']
    print('Davies-Bouldin Index ({}): {}'.format(cfg['model_tag'], dbi_result))
    save(dbi_result, './output/result/dbi_{}.npy'.format(cfg['model_tag']), mode='numpy')
    return evaluation


if __name__ == "__main__":
    main()