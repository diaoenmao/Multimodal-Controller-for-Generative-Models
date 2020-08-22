import argparse
import itertools

parser = argparse.ArgumentParser(description='Config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--num_gpu', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
args = vars(parser.parse_args())


def main():
    run = args['run']
    model = args['model']
    num_gpu = args['num_gpu']
    world_size = args['world_size']
    round = args['round']
    experiment_step = args['experiment_step']
    num_experiments = args['num_experiments']
    num_epochs = args['num_epochs']
    resume_mode = args['resume_mode']
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpu, world_size))]
    if run in ['train', 'test']:
        filename = '{}_{}'.format(run, model)
        script_name = [['{}_{}.py'.format(run, model)]]
    else:
        filename = '{}_{}'.format(run, model)
        script_name = [['{}.py'.format(run)]]
    data_names = [['CIFAR10', 'CIFAR100', 'COIL100', 'Omniglot']]
    if model == 'vqvae':
        model_names = [['vqvae']]
    else:
        model_names = [['c{}'.format(args['model']), 'mc{}'.format(args['model'])]]
    init_seeds = [list(range(0, num_experiments, experiment_step))]
    num_experiments = [[experiment_step]]
    num_epochs = [[num_epochs]]
    resume_mode = [[resume_mode]]
    world_size = [[world_size]]
    s = '#!/bin/bash\n'
    k = 0
    controls = script_name + data_names + model_names + init_seeds + world_size + num_experiments + \
               num_epochs + resume_mode
    controls = list(itertools.product(*controls))
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        if 'mc' in controls[i][2]:
            controls[i].append('0.5')
        else:
            controls[i].append('None')
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --data_name {} --model_name {} --init_seed {} ' \
                '--world_size {} --num_experiments {} --num_epochs {} --resume_mode {} --control_name {}&\n'.format(
            gpu_ids[k % len(gpu_ids)], *controls[i])
        if k % round == round - 1:
            s = s[:-2] + '\nwait\n'
        k = k + 1
    print(s)
    run_file = open('./{}.sh'.format(filename), 'w')
    run_file.write(s)
    run_file.close()
    return


if __name__ == '__main__':
    main()