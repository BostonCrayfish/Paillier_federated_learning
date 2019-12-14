# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    #### 实际上aggregator没有密钥，这里是为了查看实验结果
    # if is_paillier==True:
    #     f = open('key pr', 'rb')
    #     data = pickle.load(f)

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg



## 暴力求均值：
def Fedupdate(w_old,w):
    w_avg = w_old
    for k in w_old.keys():
        w_avg[k]=(w_old[k]+w[k])/2
    return w_avg
