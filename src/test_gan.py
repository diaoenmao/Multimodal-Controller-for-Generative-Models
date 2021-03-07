import argparse
import datetime
import os
import torch
import torch.backends.cudnn as cudnn
import models
import numpy as np
from config import cfg
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, process_control, process_dataset, resume, collate, save_img
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
cfg['control_name'] = '_'.join(
    [cfg['control'][k] for k in cfg['control'] if cfg['control'][k]]) if 'control' in cfg else ''


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    metric = Metric({'test': ['InceptionScore', 'FID']})
    last_epoch, model, _, _, _ = resume(model, cfg['model_tag'], load_tag='best')
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    logger_path = 'output/runs/test_{}_{}'.format(cfg['model_tag'], current_time)
    test_logger = Logger(logger_path)
    test_logger.safe(True)
    test(model, metric, test_logger, last_epoch)
    test_logger.safe(False)
    _, _, _, _, train_logger = resume(model, cfg['model_tag'], load_tag='checkpoint')
    save_result = {'cfg': cfg, 'epoch': last_epoch, 'logger': {'train': train_logger, 'test': test_logger}}
    save(save_result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def test(model, metric, logger, epoch):
    sample_per_iter = cfg[cfg['model_name']]['batch_size']['test']
    generate_per_mode = 1000
    with torch.no_grad():
        model.train(False)
        C = torch.arange(cfg['target_size'])
        C = C.repeat(generate_per_mode)
        z = torch.randn([C.size(0), cfg[cfg['model_name']]['latent_size']])
        z = torch.split(z, sample_per_iter)
        C = torch.split(C, sample_per_iter)
        generated = []
        for i in range(len(z)):
            z_i = z[i].to(cfg['device'])
            C_i = C[i].to(cfg['device'])
            generated_i = model.generator(z_i, C_i)
            generated.append(generated_i.cpu())
        generated = torch.cat(generated)
        output = {'data': generated}
        evaluation = metric.evaluate(metric.metric_name['test'], input, output)
        logger.append(evaluation, 'test')
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
        save_per_mode = 10
        saved = []
        for i in range(0, cfg['target_size'] * save_per_mode, cfg['target_size']):
            saved.append(generated[i:i + cfg['target_size']])
        saved = torch.cat(saved)
        save_img(saved, './output/vis/{}.png'.format(cfg['model_tag']), nrow=cfg['target_size'], range=(-1, 1))
    return


if __name__ == "__main__":
    main()
