# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
from .Nets import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    def __init__(self,net=None,train_loader=None,test_loader=None,local_ep=1):
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = train_loader
        self.ldr_test = test_loader
        self.optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.5)
        self.net = net
        self.local_ep=local_ep

    def train(self, w=None):
        net = self.net
        if w!=None:
            net.load_state_dict(w)
        net.train()
        # train and update
        epoch_loss = []
        for iter in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(device), labels.to(device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict()


def get_net_and_trainer(model_name="mlp",dataset="mnist"):
    ### To do : get part of data
    if dataset=='mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    elif dataset=='cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

    train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=128, shuffle=True)
    img_size = dataset_train[0][0].shape

    if model_name == 'cnn' and dataset == 'cifar':
        net_glob = CNNCifar().to(device)
    elif model_name == 'cnn' and dataset == 'mnist':
        net_glob = CNNMnist().to(device)
    elif model_name == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=10).to(device)
    else:
        exit('Error: unrecognized model')

    return net_glob,train_loader,test_loader

if __name__ == '__main__':
    net,train_loder,test_loader=get_net_and_trainer()
    trainer=Trainer(net,train_loder,test_loader)
    w=net.state_dict()
    w,loss=trainer.train(w)
    ## w-> aggregator -> new w
    w,loss=trainer.train(w)