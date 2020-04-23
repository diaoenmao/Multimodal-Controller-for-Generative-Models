import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import models
import numpy as np
from utils import recur, save, load, to_device
from torch.utils.data import DataLoader
from torchvision.models.inception import inception_v3
from scipy.stats import entropy

def MSE(output, target):
    with torch.no_grad():
        mse = F.mse_loss(output, target, reduction='mean').item()
    return mse


def NLL(output, target):
    with torch.no_grad():
        NLL = F.cross_entropy(output, target, reduction='mean').item()
    return NLL


def PSNR(output, target, MAX=1.0):
    with torch.no_grad():
        max = torch.tensor(MAX).to(config.PARAM['device'])
        mse = F.mse_loss(output.to(torch.float64), target.to(torch.float64))
        psnr = (20 * torch.log10(max) - 10 * torch.log10(mse)).item()
    return psnr


def InceptionScore(img, splits=10):
    N = len(img)
    batch_size = 256
    data_loader = DataLoader(img, batch_size=batch_size)
    if config.PARAM['data_name'] in ['MNIST', 'Omniglot']:
        model = eval('models.classifier().to(config.PARAM["device"])')
        model_tag = ['0', config.PARAM['data_name'], config.PARAM['subset'], 'classifier']
        model_tag = '_'.join(filter(None, model_tag))
        checkpoint = load('./metrics_tf/res/classifier/{}_best.pt'.format(model_tag))
        model.load_state_dict(checkpoint['model_dict'])
        model.train(False)
        preds = np.zeros((N, config.PARAM['classes_size']))
        for i, input in enumerate(data_loader):
            input = {'img': input, 'label': input.new_zeros(input.size(0)).long()}
            input = to_device(input, config.PARAM['device'])
            input_size_i = input['img'].size(0)
            output = model(input)
            preds[i * batch_size:i * batch_size + input_size_i] = F.softmax(output['label'], dim=-1).cpu().numpy()
    else:
        model = inception_v3(pretrained=True, transform_input=False).to(config.PARAM['device'])
        model.train(False)
        up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
        preds = np.zeros((N, 1000))
        for i, input in enumerate(data_loader):
            input = input.to(config.PARAM['device'])
            input_size_i = input.size(0)
            input = up(input)
            output = model(input)
            preds[i * batch_size:i * batch_size + input_size_i] = F.softmax(output, dim=-1).cpu().numpy()
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    is_mean = np.mean(split_scores).item()
    is_std = np.std(split_scores).item()
    return is_mean, is_std


def Accuracy(output, target, topk=1):
    with torch.no_grad():
        batch_size = target.size(0)
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.view(-1, 1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


class Metric(object):
    def __init__(self):
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'Loss_G': (lambda input, output: output['loss_G'].item()),
                       'Loss_D': (lambda input, output: output['loss_D'].item()),
                       'InceptionScore': (lambda input, output: recur(InceptionScore, output['img'])),
                       'Accuracy': (lambda input, output: recur(Accuracy, output['label'], input['label'])),
                       'MSE': (lambda input, output: recur(MSE, output['img'], input['img'])),
                       'NLL': (lambda input, output: recur(NLL, output['logits'], input['code'])),
                       'PSNR': (lambda input, output: recur(PSNR, output['img']))}

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation