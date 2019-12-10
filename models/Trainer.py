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
from progress.bar import Bar
import time

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

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            end = time.time()

            bar = Bar('Processing', max=len(self.ldr_train))

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                data_time.update(time.time() - end)
                
                images, labels = images.to(device), labels.to(device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                prec1, prec5 = accuracy(log_probs.data, labels.data, topk=(1, 5))
                losses.update(loss.data.item(), images.size(0))
                top1.update(prec1.item(), images.size(0))
                top5.update(prec5.item(), images.size(0))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_time.update(time.time() - end)
                end = time.time()

                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(self.ldr_train),
                    data=data_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
                bar.next()
            bar.finish()

            #     if batch_idx % 10 == 0:
            #         print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #             iter, batch_idx * len(images), len(self.ldr_train.dataset),
            #                    100. * batch_idx / len(self.ldr_train), loss.item()))
            #     batch_loss.append(loss.item())
            # epoch_loss.append(sum(batch_loss)/len(batch_loss))
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))


    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# -*- coding: utf-8 -*-
# Python version: 3.6

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

            



if __name__ == '__main__':
    import sys
    net,train_loder,test_loader=get_net_and_trainer()
    trainer=Trainer(net,train_loder,test_loader,local_ep=int(sys.argv[1]))
    w=net.state_dict()
    w=trainer.train(w)
    ## w-> aggregator -> new w
    w=trainer.train(w)