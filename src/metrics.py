import config
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils import recur


def NLL(output, target):
    with torch.no_grad():
        NLL = F.binary_cross_entropy_with_logits(output, target, reduction='sum').item() / output.size(0)
    return NLL


def PSNR(output, target, MAX=1.0):
    with torch.no_grad():
        max = torch.tensor(MAX).to(config.PARAM['device'])
        mse = F.mse_loss(output.to(torch.float64), target.to(torch.float64))
        psnr = (20 * torch.log10(max) - 10 * torch.log10(mse)).item()
    return psnr


class Metric(object):
    def __init__(self):
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'NLL': (lambda input, output: recur(NLL, output['img'], input['img'])),
                       'PSNR': (lambda input, output: recur(PSNR, output['img'], input['img']))}

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation