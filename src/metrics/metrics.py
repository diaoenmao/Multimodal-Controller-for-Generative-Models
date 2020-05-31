import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import models
import numpy as np
from scipy import linalg
from utils import recur, save, load, to_device, collate
from data import fetch_dataset, make_data_loader
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


def FID(img):
    batch_size = 256
    config.PARAM['batch_size']['train'] = batch_size
    dataset = fetch_dataset(config.PARAM['data_name'], config.PARAM['subset'])
    real_data_loader = make_data_loader(dataset)['train']
    generated_data_loader = DataLoader(img, batch_size=batch_size)
    model = eval('models.classifier().to(config.PARAM["device"])')
    model_tag = ['0', config.PARAM['data_name'], config.PARAM['subset'], 'classifier']
    model_tag = '_'.join(filter(None, model_tag))
    checkpoint = load('./metrics_tf/res/classifier/{}_best.pt'.format(model_tag))
    model.load_state_dict(checkpoint['model_dict'])
    model.train(False)
    real_feature = []
    for i, input in enumerate(real_data_loader):
        input = collate(input)
        input = to_device(input, config.PARAM['device'])
        real_feature_i = model.feature(input)
        real_feature.append(real_feature_i)
    real_feature = torch.cat(real_feature, dim=0).cpu().numpy()
    generated_feature = []
    for i, input in enumerate(generated_data_loader):
        input = {'img': input, 'label': input.new_zeros(input.size(0)).long()}
        input = to_device(input, config.PARAM['device'])
        generated_feature_i = model.feature(input)
        generated_feature.append(generated_feature_i)
    generated_feature = torch.cat(generated_feature, dim=0).cpu().numpy()
    mu1 = np.mean(real_feature, axis=0)
    sigma1 = np.cov(real_feature, rowvar=False)
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
    def __init__(self):
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'Loss_G': (lambda input, output: output['loss_G'].item()),
                       'Loss_D': (lambda input, output: output['loss_D'].item()),
                       'InceptionScore': (lambda input, output: recur(InceptionScore, output['img'])),
                       'FID': (lambda input, output: recur(FID, output['img'])),
                       'Accuracy': (lambda input, output: recur(Accuracy, output['label'], input['label'])),
                       'MSE': (lambda input, output: recur(MSE, output['img'], input['img'])),
                       'NLL': (lambda input, output: recur(NLL, output['logits'], input['img'])),
                       'PSNR': (lambda input, output: recur(PSNR, output['img']))}

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation