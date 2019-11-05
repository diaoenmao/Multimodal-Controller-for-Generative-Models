import config
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils import recur


def Accuracy(output, target):
    topk = config.PARAM['topk']
    with torch.no_grad():
        target_size = target.numel()
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.unsqueeze(1).expand_as(pred_k)).float().sum()
        accuracy = (correct_k * (100.0 / target_size)).item()
    return accuracy


class Metric(object):
    def __init__(self):
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'Accuracy': (lambda input, output: recur(Accuracy, output['label'], input['label']))}

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation