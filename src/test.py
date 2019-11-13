import config

config.init()
import argparse
import datetime
import torch
import torch.backends.cudnn as cudnn
import models
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, load, to_device, process_control_name, process_dataset, resume, collate
from logger import Logger

# if __name__ == "__main__":
#     test_0 = load('output/test_0.pt')
#     test_1 = load('output/test_1.pt')
#     n = 1
#     c_0 = [[] for _ in range(n)]
#     c_1 = [[] for _ in range(n)]
#     print(len(test_0), len(test_1))
#     for i in range(len(test_0)):
#         for j in range(len(test_0[i])):
#             c_0[j].append(test_0[i][j])
#     for i in range(len(test_1)):
#         for j in range(len(test_1[i])):
#             c_1[j].append(test_1[i][j])
#     for i in range(n):
#         c_0[i] = torch.cat(c_0[i], dim=0)
#         c_1[i] = torch.cat(c_1[i], dim=0)
#     for i in range(n):
#         print(torch.eq(c_0[i], c_0[i]).all())
#         print((c_0[i]-c_1[i]).abs().mean())