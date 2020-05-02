import config

config.init()
import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, load, to_device, process_control_name, process_dataset, collate
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
config.PARAM['metric_names'] = {'train': ['Loss', 'Loss_D', 'Loss_G'], 'test': ['InceptionScore']}


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
    last_epoch, model, optimizer, scheduler, logger = resume(model, config.PARAM['model_tag'])
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    logger_path = 'output/runs/test_{}_{}'.format(config.PARAM['model_tag'], current_time) if config.PARAM[
        'log_overwrite'] else 'output/runs/test_{}'.format(config.PARAM['model_tag'])
    logger = Logger(logger_path)
    logger.safe(True)
    test(model, logger, last_epoch)
    logger.safe(False)
    save_result = {
        'config': config.PARAM, 'epoch': last_epoch, 'logger': logger}
    save(save_result, './output/result/{}.pt'.format(config.PARAM['model_tag']))
    return


def test(model, logger, epoch):
    sample_per_iter = 1000
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        C = torch.arange(config.PARAM['classes_size']).to(config.PARAM['device'])
        C = C.repeat(config.PARAM['generate_per_mode'])
        config.PARAM['z'] = torch.randn([C.size(0), config.PARAM['latent_size']], device=config.PARAM['device']) \
            if 'z' not in config.PARAM else config.PARAM['z']
        C_generated = torch.split(C, sample_per_iter)
        z_generated = torch.split(config.PARAM['z'], sample_per_iter)
        generated = []
        for i in range(len(C_generated)):
            C_generated_i = C_generated[i]
            z_generated_i = z_generated[i]
            generated_i = model.generate(z_generated_i, C_generated_i)
            generated.append(generated_i)
        generated = torch.cat(generated)
        output = {'img': generated}
        evaluation = metric.evaluate(config.PARAM['metric_names']['test'], None, output)
        logger.append(evaluation, 'test')
        info = {'info': ['Model: {}'.format(config.PARAM['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', config.PARAM['metric_names']['test'])
    return


def resume(model, model_tag, optimizer=None, scheduler=None, load_tag='checkpoint'):
    if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
        checkpoint = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_dict'])
        if optimizer is not None:
            optimizer['generator'].load_state_dict(checkpoint['optimizer_dict']['generator'])
            optimizer['discriminator'].load_state_dict(checkpoint['optimizer_dict']['discriminator'])
        if scheduler is not None:
            scheduler['generator'].load_state_dict(checkpoint['scheduler_dict']['generator'])
            scheduler['discriminator'].load_state_dict(checkpoint['scheduler_dict']['discriminator'])
        logger = checkpoint['logger']
        print('Resume from {}'.format(last_epoch))
    else:
        print('Not exists model tag: {}, start from scratch'.format(model_tag))
        import datetime
        from logger import Logger
        last_epoch = 1
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/train_{}_{}'.format(config.PARAM['model_tag'], current_time) if config.PARAM[
            'log_overwrite'] else 'output/runs/train_{}'.format(config.PARAM['model_tag'])
        logger = Logger(logger_path)
    return last_epoch, model, optimizer, scheduler, logger


if __name__ == "__main__":
    main()