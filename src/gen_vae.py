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
    load_tag = 'best'
    _, model, _, _, _ = resume(model, config.PARAM['model_tag'], load_tag=load_tag)
    test(model)
    return


def test(model):
    save_per_mode = 10
    save_num_mode = min(100, config.PARAM['classes_size'])
    sample_per_iter = 1000
    with torch.no_grad():
        model.train(False)
        C = torch.arange(config.PARAM['classes_size']).to(config.PARAM['device'])
        C = C.repeat(config.PARAM['generate_per_mode'])
        config.PARAM['z'] = torch.randn(
            [C.size(0), config.PARAM['quantizer_embedding_size'], config.PARAM['img_shape'][1] // 4,
             config.PARAM['img_shape'][2] // 4], device=config.PARAM['device'])
        C_generated = torch.split(C, sample_per_iter)
        z_generated = torch.split(config.PARAM['z'], sample_per_iter)
        generated = []
        saved = []
        for i in range(len(C_generated)):
            C_generated_i = C_generated[i]
            z_generated_i = z_generated[i]
            generated_i = model.generate(C_generated_i, z_generated_i)
            generated.append(generated_i)
            saved.append(generated_i[:save_per_mode])
        generated = torch.cat(generated)
        saved = torch.cat(saved)
        generated = ((generated + 1) / 2 * 255).cpu().numpy()
        saved = (saved + 1) / 2
        save(generated, './output/npy/{}.npy'.format(config.PARAM['model_tag']), mode='numpy')
        save_img(saved, './output/img/generated_{}.png'.format(config.PARAM['model_tag']), nrow=save_num_mode)
    return


if __name__ == "__main__":
    main()