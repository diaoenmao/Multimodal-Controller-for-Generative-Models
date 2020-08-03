import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from config import cfg
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, load, to_device, process_control, process_dataset, collate, save_img
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
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
cfg['pivot_metric'] = 'InceptionScore'
cfg['pivot'] = -float('inf')
if cfg['data_name'] in ['ImageNet', 'ImageNet32']:
    cfg['batch_size'] = {'train': 1024, 'test': 1024}
else:
    cfg['batch_size'] = {'train': 128, 'test': 512}
cfg['metric_name'] = {'train': ['Loss', 'Loss_D', 'Loss_G'], 'test': ['InceptionScore']}
cfg['optimizer_name'] = 'Adam'
if cfg['model_name'] == 'cgan':
    if cfg['data_name'] in ['CIFAR10']:
        cfg['lr'] = {'generator': 2e-4, 'discriminator': 2e-4}
        cfg['iter'] = {'generator': 1, 'discriminator': 5}
        cfg['betas'] = {'generator': (0, 0.9), 'discriminator': (0, 0.9)}
    elif cfg['data_name'] in ['CIFAR100']:
        cfg['lr'] = {'generator': 2e-4, 'discriminator': 2e-4}
        cfg['iter'] = {'generator': 1, 'discriminator': 5}
        cfg['betas'] = {'generator': (0, 0.9), 'discriminator': (0, 0.9)}
    elif cfg['data_name'] in ['Omniglot']:
        cfg['lr'] = {'generator': 2e-4, 'discriminator': 2e-4}
        cfg['iter'] = {'generator': 1, 'discriminator': 5}
        cfg['betas'] = {'generator': (0.5, 0.999), 'discriminator': (0.5, 0.999)}
elif cfg['model_name'] == 'mcgan':
    if cfg['data_name'] in ['CIFAR10']:
        cfg['lr'] = {'generator': 2e-4, 'discriminator': 2e-4}
        cfg['iter'] = {'generator': 1, 'discriminator': 5}
        cfg['betas'] = {'generator': (0.5, 0.999), 'discriminator': (0.5, 0.999)}
    elif cfg['data_name'] in ['CIFAR100']:
        cfg['lr'] = {'generator': 2e-4, 'discriminator': 2e-4}
        cfg['iter'] = {'generator': 1, 'discriminator': 5}
        cfg['betas'] = {'generator': (0.5, 0.999), 'discriminator': (0.5, 0.999)}
    elif cfg['data_name'] in ['Omniglot']:
        cfg['lr'] = {'generator': 2e-4, 'discriminator': 2e-4}
        cfg['iter'] = {'generator': 1, 'discriminator': 5}
        cfg['betas'] = {'generator': (0.5, 0.999), 'discriminator': (0.5, 0.999)}
else:
    raise ValueError('Not valid model name')
cfg['weight_decay'] = 0
cfg['scheduler_name'] = 'None'
cfg['loss_type'] = 'Hinge'


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
    data_loader = make_data_loader(dataset)
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    optimizer = {'generator': make_optimizer(model.generator, cfg['lr']['generator'], cfg['betas']['generator']),
                 'discriminator': make_optimizer(model.discriminator, cfg['lr']['discriminator'],
                                                 cfg['betas']['discriminator'])}
    scheduler = {'generator': make_scheduler(optimizer['generator']),
                 'discriminator': make_scheduler(optimizer['discriminator'])}
    if cfg['resume_mode'] == 1:
        last_epoch, model, optimizer, scheduler, logger = resume(model, cfg['model_tag'], optimizer, scheduler)
    elif cfg['resume_mode'] == 2:
        last_epoch = 1
        _, model, _, _, _ = resume(model, cfg['model_tag'])
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/{}_{}'.format(cfg['model_tag'], current_time)
        logger = Logger(logger_path)
    else:
        last_epoch = 1
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], current_time)
        logger = Logger(logger_path)
    if cfg['world_size'] > 1:
        model.generator = torch.nn.DataParallel(model.generator, device_ids=list(range(cfg['world_size'])))
        model.discriminator = torch.nn.DataParallel(model.discriminator, device_ids=list(range(cfg['world_size'])))
    for epoch in range(last_epoch, cfg['num_epochs'] + 1):
        logger.safe(True)
        train(data_loader['train'], model, optimizer, logger, epoch)
        test(model, logger, epoch)
        if cfg['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler['generator'].step(metrics=logger.mean['test/{}'.format(cfg['pivot_metric'])][0])
            scheduler['discriminator'].step(metrics=logger.mean['test/{}'.format(cfg['pivot_metric'])][0])
        else:
            scheduler['generator'].step()
            scheduler['discriminator'].step()
        logger.safe(False)
        if cfg['world_size'] > 1:
            model.generator, model.discriminator = model.module.generator, model.module.discriminator
        model_state_dict = model.state_dict()
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'model_dict': model_state_dict,
            'optimizer_dict': {'generator': optimizer['generator'].state_dict(),
                               'discriminator': optimizer['discriminator'].state_dict()},
            'scheduler_dict': {'generator': scheduler['generator'].state_dict(),
                               'discriminator': scheduler['discriminator'].state_dict()}, 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if cfg['pivot'] < logger.mean['test/{}'.format(cfg['pivot_metric'])][0]:
            cfg['pivot'] = logger.mean['test/{}'.format(cfg['pivot_metric'])][0]
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(data_loader, model, optimizer, logger, epoch):
    metric = Metric()
    model.train(True)
    for i, input in enumerate(data_loader):
        start_time = time.time()
        input = collate(input)
        input_size = input['img'].size(0)
        input = to_device(input, cfg['device'])
        ############################
        # (1) Update D network
        ###########################
        for _ in range(cfg['iter']['discriminator']):
            # train with real
            optimizer['discriminator'].zero_grad()
            optimizer['generator'].zero_grad()
            D_x = model.discriminate(input['img'], input[cfg['subset']])
            # train with fake
            z1 = torch.randn(input['img'].size(0), cfg['gan']['latent_size'], device=cfg['device'])
            generated = model.generate(input[cfg['subset']], z1)
            D_G_z1 = model.discriminate(generated.detach(), input[cfg['subset']])
            if cfg['loss_type'] == 'BCE':
                D_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    D_x, torch.ones((input['img'].size(0), 1), device=cfg['device'])) + \
                         torch.nn.functional.binary_cross_entropy_with_logits(
                             D_G_z1, torch.zeros((input['img'].size(0), 1), device=cfg['device']))
            elif cfg['loss_type'] == 'Hinge':
                D_loss = torch.nn.functional.relu(1.0 - D_x).mean() + torch.nn.functional.relu(1.0 + D_G_z1).mean()
            else:
                raise ValueError('Not valid loss type')
            D_loss.backward()
            optimizer['discriminator'].step()
        ############################
        # (2) Update G network
        ###########################
        for _ in range(cfg['iter']['generator']):
            optimizer['discriminator'].zero_grad()
            optimizer['generator'].zero_grad()
            z2 = torch.randn(input['img'].size(0), cfg['gan']['latent_size'], device=cfg['device'])
            generated = model.generate(input[cfg['subset']], z2)
            D_G_z2 = model.discriminate(generated, input[cfg['subset']])
            if cfg['loss_type'] == 'BCE':
                G_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    D_G_z2, torch.ones((input['img'].size(0), 1), device=cfg['device']))
            elif cfg['loss_type'] == 'Hinge':
                G_loss = -D_G_z2.mean()
            else:
                raise ValueError('Not valid loss type')
            G_loss.backward()
            optimizer['generator'].step()
        output = {'loss': abs(D_loss - G_loss), 'loss_D': D_loss, 'loss_G': G_loss}
        evaluation = metric.evaluate(cfg['metric_name']['train'], input, output)
        logger.append(evaluation, 'train', n=input_size)
        if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
            batch_time = time.time() - start_time
            generator_lr, discriminator_lr = optimizer['generator'].param_groups[0]['lr'], \
                                             optimizer['discriminator'].param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['num_epochs'] - epoch) * batch_time * len(data_loader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                             'Learning rate : (G: {}, D: {})'.format(generator_lr, discriminator_lr),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            logger.write('train', cfg['metric_name']['train'])
    return


def test(model, logger, epoch):
    sample_per_iter = cfg['batch_size']['test']
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        C = torch.arange(cfg['classes_size'])
        C = C.repeat(cfg['generate_per_mode'])
        cfg['z'] = torch.randn([C.size(0), cfg['gan']['latent_size']]) if 'z' not in cfg else cfg['z']
        C_generated = torch.split(C, sample_per_iter)
        z_generated = torch.split(cfg['z'], sample_per_iter)
        generated = []
        for i in range(len(C_generated)):
            C_generated_i = C_generated[i].to(cfg['device'])
            z_generated_i = z_generated[i].to(cfg['device'])
            generated_i = model.generate(C_generated_i, z_generated_i)
            generated.append(generated_i.cpu())
        generated = torch.cat(generated)
        output = {'img': generated}
        evaluation = metric.evaluate(cfg['metric_name']['test'], None, output)
        logger.append(evaluation, 'test')
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test'])
    return


def make_optimizer(model, lr, betas):
    if cfg['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=cfg['momentum'],
                                  weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'], betas=betas)
    elif cfg['optimizer_name'] == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'], betas=betas)
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer):
    if cfg['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'],
                                              gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'],
                                                   gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'])
    elif cfg['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg['factor'],
                                                         patience=cfg['patience'], verbose=True,
                                                         threshold=cfg['threshold'], threshold_mode='rel',
                                                         min_lr=cfg['min_lr'])
    elif cfg['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg['lr'], max_lr=10 * cfg['lr'])
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
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], current_time)
        logger = Logger(logger_path)
    return last_epoch, model, optimizer, scheduler, logger


if __name__ == "__main__":
    main()