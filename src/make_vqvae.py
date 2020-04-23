import config

config.init()
import itertools


def main():
    round = 24
    run_mode = 'train'
    model_mode = 'vqvae'
    filename = '{}_{}'.format(run_mode, model_mode)
    gpu_ids = ['0', '1', '2', '3']
    script_name = [['{}_{}.py'.format(run_mode, model_mode)]]
    data_names = ['CIFAR10', 'Omniglot']
    model_names = [['dccvqvae', 'dcmcvqvae']]
    experiments_step = 2
    num_experiments = 12
    init_seeds = [list(range(0, num_experiments, experiments_step))]
    num_epochs = [[200]]
    num_experiments = [[experiments_step]]
    s = '#!/bin/bash\n'
    for i in range(len(data_names)):
        data_name = data_names[i]
        controls = script_name + [[data_name]] + model_names + init_seeds + num_epochs + num_experiments
        controls = list(itertools.product(*controls))
        for j in range(len(controls)):
            controls[j] = list(controls[j])
            if controls[j][2] == 'dccvqvae':
                controls[j].append('None')
            elif controls[j][2] == 'dcmcvqvae':
                controls[j].append('0.5')
            else:
                raise ValueError('Not valid model name')
            s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --data_name {} --model_name {} --init_seed {} --num_epochs ' \
                    '{} --num_experiments {} --control_name {}&\n'.format(
                gpu_ids[j % len(gpu_ids)], *controls[j])
            if j % round == round - 1:
                s = s[:-2] + '\n'
    print(s)
    run_file = open('./{}.sh'.format(filename), 'w')
    run_file.write(s)
    run_file.close()
    exit()


if __name__ == '__main__':
    main()