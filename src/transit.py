import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
import numpy as np
from config import cfg
from data import fetch_dataset
from utils import process_control, process_dataset, resume, save_img

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
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    _, model, _, _, _ = resume(model, cfg['model_tag'], load_tag='best')
    transit(model)
    return


def transit(model):
    with torch.no_grad():
        model.train(False)
        root = 0
        save_num_step = cfg['save_per_mode']
        alphas = np.linspace(0, 1, save_num_step + 1)
        max_save_num_mode = [10, 50, 100]
        for i in range(len(max_save_num_mode)):
            if max_save_num_mode[i] > cfg['classes_size']:
                continue
            save_num_mode = min(max_save_num_mode[i], cfg['classes_size'])
            C = torch.arange(save_num_mode).to(cfg['device'])
            if cfg['model_name'] in ['cvae', 'mcvae']:
                x = torch.randn([C.size(0), cfg['vae']['latent_size']]).to(cfg['device'])
            elif cfg['model_name'] in ['cgan', 'mcgan']:
                x = torch.randn([C.size(0), cfg['gan']['latent_size']]).to(cfg['device'])
            else:
                temperature = 1
                z_shapes = model.make_z_shapes()
                x = []
                for k in range(len(z_shapes)):
                    x_k = torch.randn([C.size(0), *z_shapes[k]], device=cfg['device']) * temperature
                    x.append(x_k)
            transited = []
            for j in range(len(alphas)):
                models.utils.transit(model, root, alphas[j])
                model = model.to(cfg['device'])
                transited_i = model.generate(C, x)
                transited.append(transited_i.cpu())
            transited = torch.stack(transited, dim=0)
            transited = transited.view(-1, *transited.size()[2:])
            save_img(transited, './output/vis/transited_{}_{}.{}'.format(
                cfg['model_tag'], save_num_mode, cfg['save_format']), nrow=save_num_mode, range=(-1, 1))
    return


if __name__ == "__main__":
    main()