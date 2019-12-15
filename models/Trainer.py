# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch
from .Nets import *
from progress.bar import Bar
import time
import os
import copy
from itertools import cycle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    def __init__(self,net=None,train_loader=None,test_loader=None,local_ep=1,batch_each_epoch=100):
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = cycle(train_loader)
        self.ldr_train_iter = iter(self.ldr_train)
        self.ldr_test = test_loader
        self.optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.5)
        self.net = net
        self.local_ep=local_ep
        self.batch_each_epoch = batch_each_epoch

    def train(self, w=None,logger=None):
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

            bar = Bar('Train', max=self.batch_each_epoch)
            bar.width=16

            batch_loss = []

            #for batch_idx, (images, labels) in enumerate(self.ldr_train):

            for batch_idx in range(self.batch_each_epoch):
                images, labels = next(self.ldr_train_iter)

            
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

                bar.suffix  = '({batch}/{size}) Total: {total:} | top1: {top1: .4f} | top5: {top5: .4f} | Loss: {loss:.4f} '.format(
                    batch=batch_idx + 1,
                    size=self.batch_each_epoch,
                    total=bar.elapsed_td,
                    top1=top1.avg,
                    top5=top5.avg,
                    loss=losses.avg,
                    )
                bar.next()
            bar.finish()

        return net.state_dict()

def compute_acc(net,test_loader,logger=None,time_waste=None,loss_func=nn.CrossEntropyLoss()):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    net.eval()

    end = time.time()
    bar = Bar('Test:', max=len(test_loader))
    bar.width=16
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        # compute output
        outputs = net(inputs)

        loss = loss_func(outputs, targets)
        losses.update(loss.data.item(), inputs.size(0))

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | top1: {top1: .4f} | top5: {top5: .4f} | Loss: {loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(test_loader),
                    total=bar.elapsed_td,
                    top1=top1.avg,
                    top5=top5.avg,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()
    if logger is not None:
        logger.append([top1.avg,losses.avg,time_waste])

   


def get_net_and_loader(model_name="mlp",dataset="mnist",mode="Part"):
    ### To do : get part of data ——> Done!
    ### Absolute path
    if mode=="ALL":
        if dataset=='mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = torchvision.datasets.MNIST('/home/dsjxtjc/2019211333/Paillier_federated_learning/data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = torchvision.datasets.MNIST('/home/dsjxtjc/2019211333/Paillier_federated_learning/data/mnist/', train=False, download=True, transform=trans_mnist)
        elif dataset=='cifar':
            trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = torchvision.datasets.CIFAR10('/home/dsjxtjc/2019211333/Paillier_federated_learning/data/cifar', train=True, download=True, transform=trans_cifar)
            dataset_test = torchvision.datasets.CIFAR10('/home/dsjxtjc/2019211333/Paillier_federated_learning/data/cifar', train=False, download=True, transform=trans_cifar)
    elif mode=="Part":
        if dataset=='mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = torchvision.datasets.ImageFolder(root='/home/dsjxtjc/2019211333/Paillier_federated_learning/data/mnist/train_jpg',transform=trans_mnist)
            dataset_test = torchvision.datasets.ImageFolder(root='/home/dsjxtjc/2019211333/Paillier_federated_learning/data/mnist/test_jpg',transform=trans_mnist)
        elif dataset=='cifar':
            trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = torchvision.datasets.ImageFolder(root='/home/dsjxtjc/2019211333/Paillier_federated_learning/data/cifar/train_jpg',transform=trans_cifar)
            dataset_test = torchvision.datasets.ImageFolder(root='/home/dsjxtjc/2019211333/Paillier_federated_learning/data/cifar/test_jpg',transform=trans_cifar)
        # elif dataset=='you dataname':
        #     trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #     dataset_train = torchvision.datasets.ImageFolder(root='/home/dsjxtjc/2019211333/Paillier_federated_learning/data/cifar/train_jpg',transform=trans_cifar)
        #     dataset_train = torchvision.datasets.ImageFolder(root='/home/dsjxtjc/2019211333/Paillier_federated_learning/data/cifar/test_jpg',transform=trans_cifar)
   


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
    import os
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

            
if __name__ == '__main__':
    import sys
    from models.logger import *

    dataset = sys.argv[3]
    net_glob = sys.argv[2]
    mode = sys.argv[4]

    if len(sys.argv)==6:
        net,train_loder,test_loader=get_net_and_loader(model_name=net_glob,dataset=dataset,mode=mode)
        trainer=Trainer(net,train_loder,test_loader,batch_each_epoch=int(sys.argv[5]))
    else:
        print('Error!')
        exit
    print(len(train_loder.dataset))
    w=net.state_dict()
    
    dir_name = 'checkpoint/trainer/{}-{}-{}'.format(dataset,net_glob,mode)
    if not os.path.isdir(dir_name):
        mkdir_p(dir_name)
    logger = Logger(os.path.join(dir_name, 'log.txt'), title='{}-{}'.format(dataset,net_glob))
    logger.set_names(['test Acc','test Loss','time'])
    time_start=time.time()

    for i in range(int(sys.argv[1])):
        print('Epoch:%s'%(i))
        w=trainer.train(w)
        net.load_state_dict(w)
        compute_acc(net,test_loader,logger,time.time()-time_start)
    