import config
config.init()
import itertools

def main():
    filename = 'vae'
    gpu_ids = ['0','1','2','3']
    script_name = [['train_vae.py']]
    data_names = ['MNIST', 'Omniglot', 'CUB200']
    model_names = [['vae', 'cvae']]
    init_seeds = [[0]]
    num_epochs = [[200]]
    s = '#!/bin/bash\n'
    for i in range(len(data_names)):
        data_name = data_names[i]
        if data_name == 'MNIST':
            control_names = [['none'], ['relu'], ['1000'], ['200'], ['2'],
                             ['1', '10', '100', '500', '1000', '10000', '0'], ['1']]
        elif data_name == 'Omniglot':
            control_names = [['none'], ['relu'], ['1000'], ['200'], ['2'], ['1', '5', '0'], ['1']]
        elif data_name == 'CUB200':
            control_names = [['none'], ['relu'], ['1000'], ['200'], ['2'], ['1', '5', '10', '20', '0'], ['1']]
        else:
            raise ValueError('Not valid data name')
        control_names = list(itertools.product(*control_names))
        control_names = [['_'.join(control_names[i]) for i in range(len(control_names))]]
        controls = script_name + [[data_name]] + model_names + init_seeds + num_epochs + control_names
        controls = list(itertools.product(*controls))
        for j in range(len(controls)):
            s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --data_name {} --model_name {} --init_seed {} --num_epochs {} --control_name {}&\n'.format(gpu_ids[j%len(gpu_ids)],*controls[j])
    print(s)
    run_file = open('./{}.sh'.format(filename), 'w')
    run_file.write(s)
    run_file.close()
    exit()
    
if __name__ == '__main__':
    main()