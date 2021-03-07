import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import numpy as np
from scipy import linalg
from config import cfg
from utils import recur, save, load, to_device, collate
from data import fetch_dataset, make_data_loader
from torch.utils.data import DataLoader
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
from sklearn.metrics import davies_bouldin_score
from .inception import InceptionV3


def InceptionScore(img, splits=1):
    with torch.no_grad():
        N = len(img)
        batch_size = cfg[cfg['model_name']]['batch_size']['train']
        data_loader = DataLoader(img, batch_size=batch_size)
        if cfg['data_name'] in ['CIFAR10']:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
            model = InceptionV3([block_idx]).to(cfg['device'])
            model.train(False)
            pred = []
            for i, input in enumerate(data_loader):
                input = input.to(cfg['device'])
                output = model(input)[0]
                pred.append(output.cpu())
            pred = torch.cat(pred, dim=0)
        else:
            model = models.conv().to(cfg['device'])
            model_tag = ['0', cfg['data_name'], 'conv']
            model_tag = '_'.join(filter(None, model_tag))
            checkpoint = load('./res/classifier/{}_best.pt'.format(model_tag))
            model.load_state_dict(checkpoint['model_dict'])
            model.train(False)
            pred = torch.zeros((N, cfg['target_size']))
            for i, input in enumerate(data_loader):
                input = {'data': input, 'target': input.new_zeros(input.size(0)).long()}
                input = to_device(input, cfg['device'])
                input_size_i = input['data'].size(0)
                output = model(input)
                pred[i * batch_size:i * batch_size + input_size_i] = F.softmax(output['target'], dim=-1).cpu()
        split_scores = []
        for k in range(splits):
            part = pred[k * (N // splits): (k + 1) * (N // splits), :]
            py = torch.mean(part, dim=0)
            scores = F.kl_div(py.log().view(1, -1).expand_as(part), part, reduction='batchmean').exp()
            split_scores.append(scores)
        inception_score = np.mean(split_scores).item()
    return inception_score


def FID(img):
    with torch.no_grad():
        generated_data_loader = DataLoader(img, batch_size=cfg[cfg['model_name']]['batch_size']['train'])
        if cfg['data_name'] in ['CIFAR10']:
            block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            model = InceptionV3([block_idx1]).to(cfg['device'])
            model.train(False)
            generated_feature = []
            for i, input in enumerate(generated_data_loader):
                input = input.to(cfg['device'])
                input_size_i = input.size(0)
                generated_feature_i = model(input)[0].view(input_size_i, -1)
                generated_feature.append(generated_feature_i.cpu().numpy())
            generated_feature = np.concatenate(generated_feature, axis=0)
        else:
            model = models.conv().to(cfg['device'])
            model_tag = ['0', cfg['data_name'], 'conv']
            model_tag = '_'.join(filter(None, model_tag))
            checkpoint = load('./res/classifier/{}_best.pt'.format(model_tag))
            model.load_state_dict(checkpoint['model_dict'])
            model.train(False)
            generated_feature = []
            for i, input in enumerate(generated_data_loader):
                input = {'data': input, 'target': input.new_zeros(input.size(0)).long()}
                input = to_device(input, cfg['device'])
                generated_feature_i = model.feature(input)
                generated_feature.append(generated_feature_i.cpu().numpy())
            generated_feature = np.concatenate(generated_feature, axis=0)
        mu1, sigma1 = load('./res/fid_stats/{}.pt'.format(cfg['data_name']))
        mu2 = np.mean(generated_feature, axis=0)
        sigma2 = np.cov(generated_feature, rowvar=False)
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
        diff = mu1 - mu2
        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        fid = fid.item()
    return fid


def Accuracy(output, target, topk=1):
    with torch.no_grad():
        batch_size = target.size(0)
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.view(-1, 1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'Loss_G': (lambda input, output: output['loss_G'].item()),
                       'Loss_D': (lambda input, output: output['loss_D'].item()),
                       'InceptionScore': (lambda input, output: recur(InceptionScore, output['data'])),
                       'FID': (lambda input, output: recur(FID, output['data'])),
                       'Accuracy': (lambda input, output: recur(Accuracy, output['target'], input['target'])), }

    def make_metric_name(self, metric_name):
        return metric_name

    def make_pivot(self):
        if cfg['data_name'] in ['MNIST', 'CIFAR10']:
            pivot = -float('inf')
            pivot_name = 'InceptionScore'
            pivot_direction = 'up'
        else:
            raise ValueError('Not valid data name')
        return pivot, pivot_name, pivot_direction

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return
