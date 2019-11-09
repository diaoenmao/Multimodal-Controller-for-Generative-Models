import config

config.init()
import argparse
import datetime
import torch
import torch.backends.cudnn as cudnn
import models
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control_name, process_dataset, resume, collate
from logger import Logger

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Config')
for k in config.PARAM:
    exec('parser.add_argument(\'--{0}\',default=config.PARAM[\'{0}\'], help=\'\')'.format(k))
args = vars(parser.parse_args())
for k in config.PARAM:
    if config.PARAM[k] != args[k]:
        exec('config.PARAM[\'{0}\'] = {1}'.format(k, args[k]))


def main():
    process_control_name()
    seeds = list(range(config.PARAM['init_seed'], config.PARAM['init_seed'] + config.PARAM['num_Experiments']))
    for i in range(config.PARAM['num_Experiments']):
        model_tag_list = [str(seeds[i]), config.PARAM['data_name'], config.PARAM['model_name'],
                          config.PARAM['control_name']]
        model_tag = '_'.join(filter(None, model_tag_list))
        print('Experiment: {}'.format(model_tag))
        runExperiment(model_tag)
    return


def runExperiment(model_tag):
    seed = int(model_tag.split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(config.PARAM['data_name'])
    process_dataset(dataset['train'])
    data_loader = make_data_loader(dataset)
    model = eval('models.{}().to(config.PARAM["device"])'.format(config.PARAM['model_name']))
    load_tag = 'best'
    last_epoch, model, _, _, _ = resume(model, model_tag, load_tag)
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    logger_path = 'output/runs/{}_{}'.format(model_tag, current_time)
    logger = Logger(logger_path)
    logger.safe(True)
    test(data_loader['test'], model, logger, last_epoch)
    logger.safe(False)
    model_state_dict = model.module.state_dict() if config.PARAM['world_size'] > 1 else model.state_dict()
    save_result = {
        'config': config.PARAM, 'epoch': last_epoch, 'model_dict': model_state_dict, 'logger': logger}
    save(save_result, './output/result/{}.pt'.format(model_tag))
    return


def test(data_loader, model, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        for i, input in enumerate(data_loader):
            input_size = len(input)
            input = collate(input)
            input = to_device(input, config.PARAM['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if config.PARAM['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(config.PARAM['metric_names']['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test')
        logger.write('test', config.PARAM['metric_names']['test'])
    return


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input


if __name__ == "__main__":
    main()