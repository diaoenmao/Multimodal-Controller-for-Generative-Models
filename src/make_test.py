import argparse
import itertools

parser = argparse.ArgumentParser(description='cfg')
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--num_gpu', default=1, type=int)
parser.add_argument('--round', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
args = vars(parser.parse_args())


def main():
    model = args['model']
    round = args['round']
    num_gpu = args['num_gpu']
    num_experiments = args['num_experiments']
    gpu_ids = [str(x) for x in list(range(num_gpu))]
    tf_data_names = ['CIFAR10', 'CIFAR100']
    pt_data_names = ['COIL100', 'Omniglot']
    created_data_names = tf_data_names + pt_data_names
    model_names = ['c{}'.format(args['model']), 'mc{}'.format(model)]
    control_exp = [str(x) for x in list(range(num_experiments))]
    s = '#!/bin/bash\n'
    k = 0
    for i in range(len(tf_data_names)):
        data_name = tf_data_names[i]
        script_name = './metrics_tf/inception_score_tf.py'
        controls = [control_exp, [data_name], ['label'], model_names]
        controls = list(itertools.product(*controls))
        for j in range(len(controls)):
            controls[j] = list(controls[j])
            if 'mc' in controls[j][3]:
                controls[j].append('0.5')
            model_tag = '_'.join(controls[j])
            s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} npy generated_{}&\n'.format(
                gpu_ids[k % len(gpu_ids)], script_name, model_tag)
            if k % round == round - 1:
                s = s[:-2] + '\n'
            k = k + 1
    k = 0
    for i in range(len(tf_data_names)):
        data_name = tf_data_names[i]
        script_name = './metrics_tf/fid_tf.py'
        controls = [control_exp, [data_name], ['label'], model_names]
        controls = list(itertools.product(*controls))
        for j in range(len(controls)):
            controls[j] = list(controls[j])
            if 'mc' in controls[j][3]:
                controls[j].append('0.5')
            model_tag = '_'.join(controls[j])
            s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} npy generated_{}&\n'.format(
                gpu_ids[k % len(gpu_ids)], script_name, model_tag)
            if k % round == round - 1:
                s = s[:-2] + '\n'
            k = k + 1
    k = 0
    for i in range(len(pt_data_names)):
        data_name = pt_data_names[i]
        script_name = 'test_generated.py'
        controls = [control_exp, [data_name], model_names]
        controls = list(itertools.product(*controls))
        for j in range(len(controls)):
            controls[j] = list(controls[j])
            if 'mc' in controls[j][2]:
                controls[j].append('0.5')
            else:
                controls[j].append('None')
            s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --data_name {} --model_name {} ' \
                    '--control_name {}&\n'.format(
                gpu_ids[k % len(gpu_ids)], script_name, *controls[j])
            if k % round == round - 1:
                s = s[:-2] + '\n'
            k = k + 1
    print(s)
    run_file = open('./test_generated.sh', 'w')
    run_file.write(s)
    run_file.close()
    s = '#!/bin/bash\n'
    k = 0
    model_names = ['c{}'.format(model), 'mc{}'.format(model)]
    for i in range(len(created_data_names)):
        data_name = created_data_names[i]
        script_name = 'test_created.py'
        controls = [control_exp, [data_name], model_names]
        controls = list(itertools.product(*controls))
        for j in range(len(controls)):
            controls[j] = list(controls[j])
            if 'mc' in controls[j][2]:
                controls[j].append('0.5')
            else:
                controls[j].append('None')
            s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --data_name {} --model_name {} ' \
                    '--control_name {}&\n'.format(
                gpu_ids[k % len(gpu_ids)], script_name, *controls[j])
            if k % round == round - 1:
                s = s[:-2] + '\n'
            k = k + 1
    print(s)
    run_file = open('./test_created.sh', 'w')
    run_file.write(s)
    run_file.close()
    return


if __name__ == '__main__':
    main()