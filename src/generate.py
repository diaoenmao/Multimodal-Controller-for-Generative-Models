import argparse
import torch
import torch.backends.cudnn as cudnn
import models
import os
from config import cfg
from data import fetch_dataset
from utils import save, process_control, process_dataset, resume, save_img

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
        ae_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['ae_name']]
        cfg['ae_tag'] = '_'.join([x for x in ae_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    process_dataset(dataset['train'])
    if 'pixelcnn' in cfg['model_name']:
        ae = eval('models.{}().to(cfg["device"])'.format(cfg['ae_name']))
        _, ae, _, _, _ = resume(ae, cfg['ae_tag'], load_tag='best')
    else:
        ae = None
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    _, model, _, _, _ = resume(model, cfg['model_tag'], load_tag='best')
    generate(model, ae)
    return


def generate(model, ae=None):
    with torch.no_grad():
        model.train(False)
        sample_per_iter = 1000
        if cfg['save_npy']:
            C = torch.arange(cfg['classes_size'])
            C = C.repeat(cfg['generate_per_mode'])
            C_generated = torch.split(C, sample_per_iter)
            generated = []
            for i in range(len(C_generated)):
                C_generated_i = C_generated[i].to(cfg['device'])
                if ae is None:
                    generated_i = model.generate(C_generated_i)
                else:
                    code_i = model.generate(C_generated_i)
                    generated_i = ae.decode_code(code_i)
                generated.append(generated_i.cpu())
            generated = torch.cat(generated)
            generated = ((generated + 1) / 2 * 255)
            save(generated.numpy(), './output/npy/generated_{}.npy'.format(cfg['model_tag']), mode='numpy')
            if cfg['save_img']:
                save_per_mode = cfg['save_per_mode']
                max_save_num_mode = 100
                save_num_mode = min(max_save_num_mode, cfg['classes_size'])
                saved = []
                for i in range(0, cfg['classes_size'] * save_per_mode, cfg['classes_size']):
                    saved.append(generated[i:i + save_num_mode])
                saved = torch.cat(saved)
                save_img(saved, './output/vis/generated_{}.{}'.format(cfg['model_tag'], cfg['save_format']),
                         nrow=save_num_mode, range=(0, 255))
        else:
            save_per_mode = cfg['save_per_mode']
            if cfg['classes_size'] > 10:
                max_save_num_mode = [10, 50, 100]
            else:
                max_save_num_mode = [10]
            for i in range(len(max_save_num_mode)):
                save_num_mode = min(max_save_num_mode[i], cfg['classes_size'])
                C = torch.arange(save_num_mode)
                C = C.repeat(save_per_mode)
                C_saved = torch.split(C, sample_per_iter)
                saved = []
                for j in range(len(C_saved)):
                    C_saved_i = C_saved[j].to(cfg['device'])
                    if ae is None:
                        saved_i = model.generate(C_saved_i)
                    else:
                        code_i = model.generate(C_saved_i)
                        saved_i = ae.decode_code(code_i)
                    saved.append(saved_i.cpu())
                saved = torch.cat(saved)
                save_img(saved, './output/vis/generated_{}_{}.{}'.format(
                    cfg['model_tag'], save_num_mode, cfg['save_format']), nrow=save_num_mode, range=(-1, 1))
    return


if __name__ == "__main__":
    main()