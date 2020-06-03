import config

config.init()
import argparse
import datetime
import os
import torch
import torch.backends.cudnn as cudnn
import models
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
        ae_tag_list = [str(seeds[i]), config.PARAM['data_name'], config.PARAM['subset'], config.PARAM['ae_name']]
        ae_tag_list = [x for x in ae_tag_list if x]
        config.PARAM['ae_tag'] = '_'.join(filter(None, ae_tag_list))
        print('Experiment: {}'.format(config.PARAM['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(config.PARAM['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(config.PARAM['data_name'], config.PARAM['subset'])
    process_dataset(dataset['train'])
    if 'pixelcnn' in config.PARAM['model_name']:
        ae = eval('models.{}().to(config.PARAM["device"])'.format(config.PARAM['ae_name']))
        _, ae, _, _, _ = resume(ae, config.PARAM['ae_tag'], load_tag='best')
    else:
        ae = None
    model = eval('models.{}().to(config.PARAM["device"])'.format(config.PARAM['model_name']))
    _, model, _, _, _ = resume(model, config.PARAM['model_tag'], load_tag='best')
    create(model, ae)
    return


def create(model, ae=None):
    save_format = 'pdf'
    with torch.no_grad():
        sample_per_iter = 1000
        if config.PARAM['save_npy']:
            models.utils.create(model)
            model = model.to(config.PARAM['device'])
            model.train(False)
            C = torch.arange(config.PARAM['classes_size'])
            C = C.repeat(config.PARAM['generate_per_mode'])
            C_created = torch.split(C, sample_per_iter)
            created = []
            for i in range(len(C_created)):
                C_created_i = C_created[i].to(config.PARAM['device'])
                if ae is None:
                    created_i = model.generate(C_created_i)
                else:
                    code_i = model.generate(C_created_i)
                    created_i = ae.decode(code_i)
                created.append(created_i.cpu())
            created = torch.cat(created)
            created = ((created + 1) / 2 * 255)
            save(created.numpy(), './output/npy/created_{}.npy'.format(config.PARAM['model_tag']), mode='numpy')
            if config.PARAM['save_img']:
                save_per_mode = config.PARAM['save_per_mode']
                max_save_num_mode = 100
                save_num_mode = min(max_save_num_mode, config.PARAM['classes_size'])
                saved = []
                for i in range(0, config.PARAM['classes_size'] * save_per_mode, config.PARAM['classes_size']):
                    saved.append(created[i:i + save_num_mode])
                saved = torch.cat(saved)
                saved = saved / 255
                save_img(saved, './output/img/created_{}.{}'.format(config.PARAM['model_tag'], save_format),
                         nrow=save_num_mode)
        else:
            if 'glow' in config.PARAM['model_name'] and config.PARAM['data_name'] == 'CIFAR10':
                save_per_mode = config.PARAM['save_per_mode']
                avoid_overflow = 1000
                save_num_modes = [10, 50, 100]
                for i in range(len(save_num_modes)):
                    save_num_mode = save_num_modes[i]
                    config.PARAM['classes_size'] = save_num_mode
                    models.utils.create(model)
                    model = model.to(config.PARAM['device'])
                    model.train(False)
                    C = torch.arange(save_num_mode)
                    C = C.repeat(avoid_overflow)
                    C_created = torch.split(C, sample_per_iter)
                    created = []
                    for j in range(len(C_created)):
                        C_created_i = C_created[j].to(config.PARAM['device'])
                        created_i = model.generate(C_created_i)
                        created.append(created_i.cpu())
                    created = torch.cat(created)
                    created = (created + 1) / 2
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
                    saved = saved.view(save_num_mode,-1, *saved.size()[1:]).transpose(0,1)
                    saved = saved.reshape(-1, *saved.size()[2:])
                    save_img(saved, './output/img/created_{}_{}.{}'.format(
                        config.PARAM['model_tag'], save_num_mode, save_format), nrow=save_num_mode)
            else:
                save_per_mode = config.PARAM['save_per_mode']
                save_num_modes = [10, 50, 100]
                for i in range(len(save_num_modes)):
                    save_num_mode = save_num_modes[i]
                    config.PARAM['classes_size'] = save_num_mode
                    models.utils.create(model)
                    model = model.to(config.PARAM['device'])
                    model.train(False)
                    C = torch.arange(save_num_mode)
                    C = C.repeat(save_per_mode)
                    C_created = torch.split(C, sample_per_iter)
                    created = []
                    for j in range(len(C_created)):
                        C_created_i = C_created[j].to(config.PARAM['device'])
                        if ae is None:
                            created_i = model.generate(C_created_i)
                        else:
                            code_i = model.generate(C_created_i)
                            created_i = ae.decode(code_i)
                        created.append(created_i.cpu())
                    created = torch.cat(created)
                    created = (created + 1) / 2
                    save_img(created, './output/img/created_{}_{}.{}'.format(
                        config.PARAM['model_tag'], save_num_mode, save_format), nrow=save_num_mode)
    return


if __name__ == "__main__":
    main()