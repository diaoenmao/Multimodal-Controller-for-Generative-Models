import config

config.init()
import argparse
import datetime
import torch
import torch.backends.cudnn as cudnn
import models
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control_name, process_dataset, resume, collate, save_img
from logger import Logger

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
    control_list = list(config.PARAM['control'].keys())
    control_name_list = args['control_name'].split('_')
    for i in range(len(control_name_list)):
        config.PARAM['control'][control_list[i]] = control_name_list[i]
control_name_list = []
for k in config.PARAM['control']:
    control_name_list.append(config.PARAM['control'][k])
config.PARAM['control_name'] = '_'.join(control_name_list)
config.PARAM['control_name'] = '_'.join(control_name_list)
config.PARAM['lr'] = 2e-4
config.PARAM['batch_size']['train'] = 64
config.PARAM['metric_names'] = {'train': ['Loss', 'Loss_D', 'Loss_G'],
                                'test': ['Loss', 'Loss_D', 'Loss_G', 'InceptionScore']}
if config.PARAM['data_name'] == 'CelebA':
    config.PARAM['subset'] = 'attr'


def main():
    process_control_name()
    seeds = list(range(config.PARAM['init_seed'], config.PARAM['init_seed'] + config.PARAM['num_Experiments']))
    for i in range(config.PARAM['num_Experiments']):
        model_tag_list = [str(seeds[i]), config.PARAM['data_name'], config.PARAM['subset'], config.PARAM['model_name'],
                          config.PARAM['control_name']]
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
    data_loader = make_data_loader(dataset)
    model = eval('models.{}().to(config.PARAM["device"])'.format(config.PARAM['model_name']))
    load_tag = 'best'
    last_epoch, model, _, _, _ = resume(model, config.PARAM['model_tag'], load_tag=load_tag)
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    logger_path = 'output/runs/test_{}_{}'.format(config.PARAM['model_tag'], current_time) if config.PARAM[
        'log_overwrite'] else 'output/runs/test_{}'.format(config.PARAM['model_tag'])
    logger = Logger(logger_path)
    logger.safe(True)
    test(data_loader['test'], model, logger, last_epoch)
    logger.safe(False)
    save_result = {
        'config': config.PARAM, 'epoch': last_epoch, 'logger': logger}
    save(save_result, './output/result/{}.pt'.format(config.PARAM['model_tag']))
    return


def test(data_loader, model, logger, epoch):
    sample_per_mode = 1000
    save_per_mode = 10
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = len(input['img'])
            input = to_device(input, config.PARAM['device'])
            input['real'] = torch.ones(input['img'].size(0), requires_grad=False, device=config.PARAM['device'])
            input['fake'] = torch.zeros(input['img'].size(0), requires_grad=False, device=config.PARAM['device'])
            D_x = model.discriminate(input['img']) if config.PARAM['model_name'] in ['gan', 'dcgan'] else \
                model.discriminate(input['img'], input[config.PARAM['subset']])
            D_x_loss = criterion(D_x, input['real'])
            generated = model.generate(input['img'].size(0)) if config.PARAM['model_name'] in ['gan', 'dcgan'] else \
                model.generate(input[config.PARAM['subset']])
            D_G_z1 = model.discriminate(generated.detach()) if config.PARAM['model_name'] in ['gan', 'dcgan'] else \
                model.discriminate(generated.detach(), input[config.PARAM['subset']])
            D_G_z1_loss = criterion(D_G_z1, input['fake'])
            generated = model.generate(input['img'].size(0)) \
                if config.PARAM['model_name'] in ['gan', 'dcgan'] else model.generate(input[config.PARAM['subset']])
            D_G_z2 = model.discriminate(generated) if config.PARAM['model_name'] in ['gan', 'dcgan'] else \
                model.discriminate(generated, input[config.PARAM['subset']])
            D_G_z2_loss = criterion(D_G_z2, input['real'])
            output = {'loss': abs((D_x_loss + D_G_z1_loss) - D_G_z2_loss), 'loss_D': D_x_loss + D_G_z1_loss,
                      'loss_G': D_G_z2_loss}
            evaluation = metric.evaluate(config.PARAM['metric_names']['test'][:-1], input, output)
            logger.append(evaluation, 'test', input_size)
        save_img(input['img'][:100],
                 './output/img/input_{}.png'.format(config.PARAM['model_tag']))
        if config.PARAM['model_name'] in ['gan', 'dcgan']:
            generated = model.generate(sample_per_mode * config.PARAM['classes_size'])
            save_img(model.generate(save_per_mode * config.PARAM['classes_size']),
                     './output/img/generated_{}.png'.format(config.PARAM['model_tag']),
                     nrow=config.PARAM['classes_size'])
        elif config.PARAM['model_name'] in ['cgan', 'dccgan', 'rmgan', 'dcrmgan']:
            generated = model.generate(
                torch.arange(config.PARAM['classes_size']).to(config.PARAM['device']).repeat(sample_per_mode))
            save_img(model.generate(
                torch.arange(config.PARAM['classes_size']).to(config.PARAM['device']).repeat(save_per_mode)),
                './output/img/generated_{}.png'.format(config.PARAM['model_tag']),
                nrow=config.PARAM['classes_size'])
        else:
            raise ValueError('Not valid model name')
        output = {'img': generated}
        evaluation = metric.evaluate(['InceptionScore'], None, output)
        logger.append(evaluation, 'test', 1)
        info = {'info': ['Model: {}'.format(config.PARAM['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', config.PARAM['metric_names']['test'])
    return


if __name__ == "__main__":
    main()