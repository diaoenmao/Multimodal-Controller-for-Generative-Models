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
config.PARAM['lr'] = 2e-4
config.PARAM['batch_size']['train'] = 64
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
    data_loader = make_data_loader(dataset)
    model = eval('models.{}().to(config.PARAM["device"])'.format(config.PARAM['model_name']))
    if config.PARAM['world_size'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(config.PARAM['world_size'])))
    optimizer = {'generator': make_optimizer(model.model['generator']),
                 'discriminator': make_optimizer(model.model['discriminator'])}
    scheduler = {'generator': make_scheduler(optimizer['generator']),
                 'discriminator': make_scheduler(optimizer['discriminator'])}
    if config.PARAM['resume_mode'] == 1:
        last_epoch, model, optimizer, scheduler, logger = resume(model, config.PARAM['model_tag'], optimizer, scheduler)
    elif config.PARAM['resume_mode'] == 2:
        last_epoch = 1
        _, model, _, _, _ = resume(model, config.PARAM['model_tag'])
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/{}_{}'.format(config.PARAM['model_tag'], current_time)
        logger = Logger(logger_path)
    else:
        last_epoch = 1
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/train_{}_{}'.format(config.PARAM['model_tag'], current_time) if config.PARAM[
            'log_overwrite'] else 'output/runs/train_{}'.format(config.PARAM['model_tag'])
        logger = Logger(logger_path)
    config.PARAM['pivot_metric'] = 'test/InceptionScore'
    config.PARAM['pivot'] = -1e10
    for epoch in range(last_epoch, config.PARAM['num_epochs'] + 1):
        logger.safe(True)
        train(data_loader['train'], model, optimizer, logger, epoch)
        test(model, logger, epoch)
        if config.PARAM['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler['generator'].step(metrics=logger.tracker[config.PARAM['pivot_metric']], epoch=epoch)
            scheduler['discriminator'].step(metrics=logger.tracker[config.PARAM['pivot_metric']], epoch=epoch)
        else:
            scheduler['generator'].step()
            scheduler['discriminator'].step()
        if config.PARAM['save_mode'] >= 0:
            logger.safe(False)
            model_state_dict = model.module.state_dict() if config.PARAM['world_size'] > 1 else model.state_dict()
            save_result = {
                'config': config.PARAM, 'epoch': epoch + 1, 'model_dict': model_state_dict,
                'optimizer_dict': {'generator': optimizer['generator'].state_dict(),
                                   'discriminator': optimizer['discriminator'].state_dict()},
                'scheduler_dict': {'generator': scheduler['generator'].state_dict(),
                                   'discriminator': scheduler['discriminator'].state_dict()}, 'logger': logger}
            save(save_result, './output/model/{}_checkpoint.pt'.format(config.PARAM['model_tag']))
            if config.PARAM['pivot'] < logger.mean[config.PARAM['pivot_metric']][0]:
                config.PARAM['pivot'] = logger.mean[config.PARAM['pivot_metric']][0]
                shutil.copy('./output/model/{}_checkpoint.pt'.format(config.PARAM['model_tag']),
                            './output/model/{}_best.pt'.format(config.PARAM['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(data_loader, model, optimizer, logger, epoch):
    criterion = torch.nn.BCELoss()
    metric = Metric()
    model.train(True)
    for i, input in enumerate(data_loader):
        start_time = time.time()
        input = collate(input)
        input_size = len(input['img'])
        input = to_device(input, config.PARAM['device'])
        input['real'] = torch.ones(input['img'].size(0), requires_grad=False, device=config.PARAM['device'])
        input['fake'] = torch.zeros(input['img'].size(0), requires_grad=False, device=config.PARAM['device'])
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        optimizer['discriminator'].zero_grad()
        D_x = model.discriminate(input['img']) if config.PARAM['model_name'] in ['gan', 'dcgan'] else \
            model.discriminate(input['img'], input[config.PARAM['subset']])
        D_x_loss = criterion(D_x, input['real'])
        D_x_loss.backward()
        # train with fake
        generated = model.generate(input['img'].size(0)) if config.PARAM['model_name'] in ['gan', 'dcgan'] else \
            model.generate(input[config.PARAM['subset']])
        D_G_z1 = model.discriminate(generated.detach()) if config.PARAM['model_name'] in ['gan', 'dcgan'] else \
            model.discriminate(generated.detach(), input[config.PARAM['subset']])
        D_G_z1_loss = criterion(D_G_z1, input['fake'])
        D_G_z1_loss.backward()
        optimizer['discriminator'].step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizer['generator'].zero_grad()
        generated = model.generate(input['img'].size(0)) \
            if config.PARAM['model_name'] in ['gan', 'dcgan'] else model.generate(input[config.PARAM['subset']])
        D_G_z2 = model.discriminate(generated) if config.PARAM['model_name'] in ['gan', 'dcgan'] else \
            model.discriminate(generated, input[config.PARAM['subset']])
        D_G_z2_loss = criterion(D_G_z2, input['real'])
        D_G_z2_loss.backward()
        optimizer['generator'].step()
        output = {'loss': abs((D_x_loss + D_G_z1_loss) - D_G_z2_loss), 'loss_D': D_x_loss + D_G_z1_loss,
                  'loss_G': D_G_z2_loss}
        if i % int((len(data_loader) * config.PARAM['log_interval']) + 1) == 0:
            batch_time = time.time() - start_time
            lr = optimizer['generator'].param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((config.PARAM['num_epochs'] - epoch) * batch_time * len(data_loader)))
            info = {'info': ['Model: {}'.format(config.PARAM['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                             'Learning rate: {}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            evaluation = metric.evaluate(config.PARAM['metric_names']['train'], input, output)
            logger.append(evaluation, 'train', n=input_size)
            logger.write('train', config.PARAM['metric_names']['train'])
    return


def test(model, logger, epoch):
    sample_per_iter = 1000
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        C = torch.arange(config.PARAM['classes_size']).to(config.PARAM['device'])
        C = C.repeat(config.PARAM['generate_per_mode'])
        config.PARAM['z'] = torch.randn([C.size(0), config.PARAM['latent_size'], 1, 1], device=config.PARAM['device']) \
            if 'z' not in config.PARAM else config.PARAM['z']
        C_generated = torch.split(C, sample_per_iter)
        z_generated = torch.split(config.PARAM['z'], sample_per_iter)
        generated = []
        for i in range(len(C_generated)):
            C_generated_i = C_generated[i]
            z_generated_i = z_generated[i]
            generated_i = model.generate(C_generated_i, z_generated_i)
            generated.append(generated_i)
        generated = torch.cat(generated)
        output = {'img': generated}
        evaluation = metric.evaluate(config.PARAM['metric_names']['test'], None, output)
        logger.append(evaluation, 'test')
        info = {'info': ['Model: {}'.format(config.PARAM['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', config.PARAM['metric_names']['test'])
    return


def make_optimizer(model):
    if config.PARAM['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.PARAM['lr'], momentum=config.PARAM['momentum'],
                              weight_decay=config.PARAM['weight_decay'])
    elif config.PARAM['optimizer_name'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config.PARAM['lr'], momentum=config.PARAM['momentum'],
                                  weight_decay=config.PARAM['weight_decay'])
    elif config.PARAM['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.PARAM['lr'], weight_decay=config.PARAM['weight_decay'],
                               betas=(0.5, 0.999))
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer):
    if config.PARAM['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif config.PARAM['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.PARAM['step_size'],
                                              gamma=config.PARAM['factor'])
    elif config.PARAM['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.PARAM['milestones'],
                                                   gamma=config.PARAM['factor'])
    elif config.PARAM['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.PARAM['num_epochs'])
    elif config.PARAM['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.PARAM['factor'],
                                                         patience=config.PARAM['patience'], verbose=True,
                                                         threshold=config.PARAM['threshold'],
                                                         threshold_mode='rel')
    elif config.PARAM['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.PARAM['lr'], max_lr=10 * config.PARAM['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


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