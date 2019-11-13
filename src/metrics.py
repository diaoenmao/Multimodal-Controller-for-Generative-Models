import config
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils import recur


def PSNR(output, target, MAX=1.0):
    with torch.no_grad():
        max = torch.tensor(MAX).to(config.PARAM['device'])
        mse = F.mse_loss(output.to(torch.float64), target.to(torch.float64))
        psnr = (20 * torch.log10(max) - 10 * torch.log10(mse)).item()
    return psnr


def BPS(code, input):
    with torch.no_grad():
        code = code.detach().cpu().numpy()
        bit_depth = math.log2(config.PARAM['num_embedding'])
        num_bit = bit_depth * code.size
        length = input.size(-1) / config.PARAM['sr']
        bps = num_bit / length
    return bps


class Metric(object):
    def __init__(self):
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'PSNR': (lambda input, output: recur(PSNR, output['img'], input['img'])),
                       'BPS': (lambda input, output: recur(BPS, output['code'], input['img']))}

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation