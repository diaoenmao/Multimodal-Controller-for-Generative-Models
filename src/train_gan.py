import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
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
cfg['control_name'] = '_'.join(
    [cfg['control'][k] for k in cfg['control'] if cfg['control'][k]]) if 'control' in cfg else ''


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    data_loader = make_data_loader(dataset, cfg['model_name'])
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    optimizer = {'generator': make_optimizer(model.generator, cfg['model_name']),
                 'discriminator': make_optimizer(model.discriminator, cfg['model_name'])}
    scheduler = {'generator': make_scheduler(optimizer['generator'], cfg['model_name']),
                 'discriminator': make_scheduler(optimizer['discriminator'], cfg['model_name'])}
    metric = Metric({'train': ['Loss_D', 'Loss_G'], 'test': ['InceptionScore', 'FID']})
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
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        logger.safe(True)
        if cfg['world_size'] > 1:
            model.generator = torch.nn.DataParallel(model.generator, device_ids=list(range(cfg['world_size'])))
            model.discriminator = torch.nn.DataParallel(model.discriminator, device_ids=list(range(cfg['world_size'])))
        train(data_loader['train'], model, optimizer, metric, logger, epoch)
        test(model, metric, logger, epoch)
        if cfg[cfg['model_name']]['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler['generator'].step(metrics=logger.mean['test/{}'.format(cfg['pivot_metric'])])
            scheduler['discriminator'].step(metrics=logger.mean['test/{}'.format(cfg['pivot_metric'])])
        else:
            scheduler['generator'].step()
            scheduler['discriminator'].step()
        if cfg['world_size'] > 1:
            model.generator, model.discriminator = model.generator.module, model.discriminator.module
        logger.safe(False)
        model_state_dict = model.state_dict()
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'model_dict': model_state_dict,
            'optimizer_dict': {'generator': optimizer['generator'].state_dict(),
                               'discriminator': optimizer['discriminator'].state_dict()},
            'scheduler_dict': {'generator': scheduler['generator'].state_dict(),
                               'discriminator': scheduler['discriminator'].state_dict()}, 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(data_loader, model, optimizer, metric, logger, epoch):
    model.train(True)
    start_time = time.time()
    for i, input in enumerate(data_loader):
        input = collate(input)
        input_size = input['data'].size(0)
        input = to_device(input, cfg['device'])
        ############################
        # (1) Update D network
        ###########################
        real = input['data']
        optimizer['discriminator'].zero_grad()
        z = torch.randn(real.size(0), cfg[cfg['model_name']]['latent_size'], device=cfg['device'])
        fake = model.generator(z, input['target'])
        real_validity = model.discriminator(real.detach(), input['target'])
        fake_validity = model.discriminator(fake.detach(), input['target'])
        D_loss = models.discriminator_loss_fn(real_validity, fake_validity, model.discriminator, real.detach(),
                                              fake.detach(), input['target'])
        D_loss.backward()
        optimizer['discriminator'].step()
        ############################
        # (2) Update G network
        ###########################
        if i % cfg[cfg['model_name']]['num_critic'] == 0:
            optimizer['generator'].zero_grad()
            z = torch.randn(real.size(0), cfg[cfg['model_name']]['latent_size'], device=cfg['device'])
            fake = model.generator(z, input['target'])
            fake_validity = model.discriminator(fake, input['target'])
            G_loss = models.generator_loss_fn(fake_validity)
            G_loss.backward()
            optimizer['generator'].step()
        output = {'loss_D': D_loss, 'loss_G': G_loss}
        evaluation = metric.evaluate(metric.metric_name['train'], input, output)
        logger.append(evaluation, 'train', n=input_size)
        if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
            batch_time = (time.time() - start_time) / (i + 1)
            generator_lr, discriminator_lr = optimizer['generator'].param_groups[0]['lr'], \
                                             optimizer['discriminator'].param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * batch_time * len(data_loader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                             'Learning rate : (G: {:.6f}, D: {:.6f})'.format(generator_lr, discriminator_lr),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    return


def test(model, metric, logger, epoch):
    sample_per_iter = cfg[cfg['model_name']]['batch_size']['test']
    generate_per_mode = 1000
    with torch.no_grad():
        model.train(False)
        C = torch.arange(cfg['target_size'])
        C = C.repeat(generate_per_mode)
        cfg['z'] = torch.randn([C.size(0), cfg[cfg['model_name']]['latent_size']]) if 'z' not in cfg else cfg['z']
        z = torch.split(cfg['z'], sample_per_iter)
        C = torch.split(C, sample_per_iter)
        generated = []
        for i in range(len(z)):
            z_i = z[i].to(cfg['device'])
            C_i = C[i].to(cfg['device'])
            generated_i = model.generator(z_i, C_i)
            generated.append(generated_i.cpu())
        generated = torch.cat(generated)
        output = {'data': generated}
        evaluation = metric.evaluate(metric.metric_name['test'], input, output)
        logger.append(evaluation, 'test')
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
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
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], current_time)
        logger = Logger(logger_path)
    return last_epoch, model, optimizer, scheduler, logger


if __name__ == "__main__":
    main()
