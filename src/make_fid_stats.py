import os
import torch
import numpy as np
import models
from config import cfg
from data import fetch_dataset, make_data_loader
from utils import save, load, collate, Stats, makedir_exist_ok, process_control, process_dataset, to_device
from metrics import InceptionV3

if __name__ == "__main__":
    stats_path = './res/fid_stats'
    data_names = ['MNIST', 'CIFAR10']
    process_control()
    cfg['seed'] = 0
    with torch.no_grad():
        for data_name in data_names:
            dataset = fetch_dataset(data_name)
            process_dataset(dataset)
            data_loader = make_data_loader(dataset, cfg['model_name'])['train']
            if data_name in ['CIFAR10']:
                print(data_name)
                block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
                model = InceptionV3([block_idx1]).to(cfg['device'])
                model.train(False)
                feature = []
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    input_size_i = input['data'].size(0)
                    feature_i = model(input['data'])[0].view(input_size_i, -1)
                    feature.append(feature_i.cpu().numpy())
                feature = np.concatenate(feature, axis=0)
            else:
                model = models.conv().to(cfg['device'])
                model_tag = ['0', cfg['data_name'], 'conv']
                model_tag = '_'.join(filter(None, model_tag))
                checkpoint = load('./res/classifier/{}_best.pt'.format(model_tag))
                model.load_state_dict(checkpoint['model_dict'])
                model.train(False)
                feature = []
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    feature_i = model.feature(input)
                    feature.append(feature_i.cpu().numpy())
                feature = np.concatenate(feature, axis=0)
            mu = np.mean(feature, axis=0)
            sigma = np.cov(feature, rowvar=False)
            stats = (mu, sigma)
            print(data_name, stats)
            makedir_exist_ok(stats_path)
            save(stats, os.path.join(stats_path, '{}.pt'.format(data_name)))
