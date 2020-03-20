import config

config.init()
import itertools


def main():
    round = 20
    run_mode = 'train'
    model_mode = 'gan'
    filename = '{}_c{}'.format(run_mode, model_mode)
    gpu_ids = ['0', '1', '2', '3']
    script_name = [['{}_{}.py'.format(run_mode, model_mode)]]
    data_names = ['MNIST', 'Omniglot']
    if model_mode == 'vae':
        model_names = [['cvae', 'dcvae']]
    else:
        model_names = [['cgan', 'dcgan']]
    init_seeds = [[0]]
    num_epochs = [[200]]
    s = '#!/bin/bash\n'
    for i in range(len(data_names)):
        data_name = data_names[i]
        if data_name in ['MNIST', 'FashionMNIST']:
            control_names = [['1', '10', '100', '500', '1000', '0']]
        elif data_name == 'Omniglot':
            control_names = [['1', '0']]
        else:
            raise ValueError('Not valid data name')
        control_names = list(itertools.product(*control_names))
        control_names = [['_'.join(control_names[i]) for i in range(len(control_names))]]
        controls = script_name + [[data_name]] + model_names + init_seeds + num_epochs + control_names
        controls = list(itertools.product(*controls))
        for j in range(len(controls)):
            s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --data_name {} --model_name {} --init_seed {} --num_epochs ' \
                    '{} --control_name {}&\n'.format(
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