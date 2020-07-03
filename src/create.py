import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
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
    create(model, ae)
    return


def create(model, ae=None):
    with torch.no_grad():
        sample_per_iter = 1000
        if cfg['save_npy']:
            models.utils.create(model)
            model = model.to(cfg['device'])
            model.train(False)
            C = torch.arange(cfg['classes_size'])
            C = C.repeat(cfg['generate_per_mode'])
            C_created = torch.split(C, sample_per_iter)
            created = []
            for i in range(len(C_created)):
                C_created_i = C_created[i].to(cfg['device'])
                if ae is None:
                    created_i = model.generate(C_created_i)
                else:
                    code_i = model.generate(C_created_i)
                    created_i = ae.decode_code(code_i)
                created.append(created_i.cpu())
            created = torch.cat(created)
            created = ((created + 1) / 2 * 255)
            save(created.numpy(), './output/npy/created_{}.npy'.format(cfg['model_tag']), mode='numpy')
            if cfg['save_img']:
                save_per_mode = cfg['save_per_mode']
                max_save_num_mode = 100
                save_num_mode = min(max_save_num_mode, cfg['classes_size'])
                saved = []
                for i in range(0, cfg['classes_size'] * save_per_mode, cfg['classes_size']):
                    saved.append(created[i:i + save_num_mode])
                saved = torch.cat(saved)
                save_img(saved, './output/img/created_{}.{}'.format(cfg['model_tag'], cfg['save_format']),
                         nrow=save_num_mode, range=(0, 255))
        else:
            if 'glow' in cfg['model_name'] and cfg['data_name'] == 'CIFAR10':
                save_per_mode = cfg['save_per_mode']
                avoid_overflow = 1000
                save_num_modes = [10, 50, 100]
                for i in range(len(save_num_modes)):
                    save_num_mode = save_num_modes[i]
                    cfg['classes_size'] = save_num_mode
                    models.utils.create(model)
                    model = model.to(cfg['device'])
                    model.train(False)
                    C = torch.arange(save_num_mode)
                    C = C.repeat(avoid_overflow)
                    C_created = torch.split(C, sample_per_iter)
                    created = []
                    for j in range(len(C_created)):
                        C_created_i = C_created[j].to(cfg['device'])
                        created_i = model.generate(C_created_i)
                        created.append(created_i.cpu())
                    created = torch.cat(created)
                    saved = []
                    for j in range(save_num_mode):
                        created_i = created[j:created.size(0):save_num_mode]
                        valid_mask = torch.sum(torch.isnan(created_i), dim=(1, 2, 3)) == 0
                        valid_created_i = created_i[valid_mask]
                        not_valid_created_i = created_i[~valid_mask]
                        created_i = valid_created_i[:save_per_mode]
                        created_i = torch.cat([created_i,
                                               not_valid_created_i[:max(save_per_mode - created_i.size(0), 0)]], dim=0)
                        saved.append(created_i)
                    saved = torch.cat(saved)
                    saved = saved.view(save_num_mode, -1, *saved.size()[1:]).transpose(0, 1)
                    saved = saved.reshape(-1, *saved.size()[2:])
                    save_img(saved, './output/img/created_{}_{}.{}'.format(
                        cfg['model_tag'], save_num_mode, cfg['save_format']), nrow=save_num_mode, range=(-1, 1))
            else:
                save_per_mode = cfg['save_per_mode']
                save_num_modes = [10, 50, 100]
                for i in range(len(save_num_modes)):
                    save_num_mode = save_num_modes[i]
                    cfg['classes_size'] = save_num_mode
                    models.utils.create(model)
                    model = model.to(cfg['device'])
                    model.train(False)
                    C = torch.arange(save_num_mode)
                    C = C.repeat(save_per_mode)
                    C_created = torch.split(C, sample_per_iter)
                    created = []
                    for j in range(len(C_created)):
                        C_created_i = C_created[j].to(cfg['device'])
                        if ae is None:
                            created_i = model.generate(C_created_i)
                        else:
                            code_i = model.generate(C_created_i)
                            created_i = ae.decode_code(code_i)
                        created.append(created_i.cpu())
                    created = torch.cat(created)
                    save_img(created, './output/img/created_{}_{}.{}'.format(
                        cfg['model_tag'], save_num_mode, cfg['save_format']), nrow=save_num_mode, range=(-1, 1))
    return


if __name__ == "__main__":
    main()