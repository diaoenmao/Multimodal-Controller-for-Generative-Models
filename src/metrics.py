import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import numpy as np
from utils import recur, collate, to_device, resume
from scipy.stats import entropy
from torch.utils.data import DataLoader
from torchvision.models.inception import inception_v3


def NLL(output, target):
    with torch.no_grad():
        NLL = F.binary_cross_entropy(output, target, reduction='sum').item() / output.size(0)
    return NLL


def PSNR(output, target, MAX=1.0):
    with torch.no_grad():
        max = torch.tensor(MAX).to(config.PARAM['device'])
        mse = F.mse_loss(output.to(torch.float64), target.to(torch.float64))
        psnr = (20 * torch.log10(max) - 10 * torch.log10(mse)).item()
    return psnr


def InceptionScore(img, batch_size=32, splits=1):
    N = len(img)
    data_loader = DataLoader(img, batch_size=batch_size)
    model = inception_v3(pretrained=True, transform_input=False).to(config.PARAM['device'])
    model.train(False)

    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)

    def get_pred(x):
        x = up(x)
        x = model(x)
        return F.softmax(x, dim=-1).data.cpu().numpy()

    preds = np.zeros((N, 1000))
    for i, input in enumerate(data_loader):
        input = input.to(config.PARAM['device'])
        input_size_i = input.size(0)
        preds[i * batch_size:i * batch_size + input_size_i] = get_pred(input)
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores).item()


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
                       'NLL': (lambda input, output: recur(NLL, output['img'], input['img'])),
                       'PSNR': (lambda input, output: recur(PSNR, output['img']))}

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation