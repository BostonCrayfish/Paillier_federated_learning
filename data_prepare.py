#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from common import *
from models.Trainer import *
dataset='cifar'
class_num=10


import sys
part_num = int(sys.argv[1])


net_glob,train_loder,test_loader=get_net_and_loader(dataset=dataset,mode="ALL")

traindata=train_loder.dataset.train_data
targetdata =train_loder.dataset.train_labels

import os
from PIL import Image
from tqdm import tqdm

def show(x,i):
    if dataset=='mnist':
        data=x.cpu().numpy()
    elif dataset=='cifar':
        data=x
    im = Image.fromarray(data)
    im.save(i)
    
## all data
dir_name='data/{}/all_train_jpg'.format(dataset)
    
for i in range(class_num):
    child_dir_name='{}/{}/'.format(dir_name,i)
    if not os.path.isdir(child_dir_name):
        mkdir_p(child_dir_name)
    
for i in tqdm(range(len(traindata))):
    #show(traindata[i],'{}/{}/{}_{}.jpg'.format(dir_name,targetdata[i],targetdata[i],i))
    pass
    # Have Done !


tdata=test_loader.dataset.test_data
ttarget =test_loader.dataset.test_labels

dir_name='data/{}/all_test_jpg'.format(dataset)
    
for i in range(class_num):
    child_dir_name='{}/{}/'.format(dir_name,i)
    if not os.path.isdir(child_dir_name):
        mkdir_p(child_dir_name)
    
for i in tqdm(range(len(tdata))):
    #show(tdata[i],'{}/{}/{}_{}.jpg'.format(dir_name,ttarget[i],ttarget[i],i))
    pass
    # Have Done !


## part data
from models.sampling import *

class DatasetSplit(Dataset):
    def __init__(self, traindata,targetdata, idxs):
        self.traindata = traindata
        self.targetdata= targetdata
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        #print(self.idxs[item])
        image, label = self.traindata[self.idxs[item]],self.targetdata[self.idxs[item]]
        return image, label
    
def save(dataset,dir_name):
    if not os.path.isdir(dir_name):
        mkdir_p(dir_name)
    
    for i in range(class_num):
        child_dir_name='{}/{}/'.format(dir_name,i)
        if not os.path.isdir(child_dir_name):
            mkdir_p(child_dir_name)
    
    for i in tqdm(range(len(dataset))):
        inputs,targets=dataset[i]

        if dataset=='mnist':
            data=inputs.cpu().numpy()
        else:
            data=inputs
            
        im = Image.fromarray(data)
        path='{}/{}/{}_{}.jpg'.format(dir_name,targets,targets,i)
        im.save(path)

dict_users = mnist_iid(traindata, part_num)

import os
cur_dir = os.getcwd()

command_send_codes = 'rm -r data/{}/part_train_jpg_{}'.format(dataset,str(part_num))
os.system(command_send_codes)

for idx in dict_users:
    x=DatasetSplit(traindata,targetdata, dict_users[idx])
    dir_name='data/{}/part_train_jpg_{}/part_{}'.format(dataset,part_num,idx+1)
 
    save(x,dir_name)
    #Have done!
    send_dir_name = 'data/{}/train_jpg'.format(dataset)
    # command_send_codes = 'ssh %s "rm -r %s/%s "; scp -r %s/%s %s:%s/%s > /dev/null &' % (host_list[idx], 
    #     cur_dir, send_dir_name,cur_dir,dir_name,host_list[idx], cur_dir,send_dir_name)
    command_send_codes = 'ssh %s "mkdir -p %s/%s "; scp -r %s/%s %s:%s/%s > /dev/null &' % (host_list[idx], 
        cur_dir, send_dir_name,cur_dir,dir_name,host_list[idx], cur_dir,send_dir_name)
    print(command_send_codes)
    os.system(command_send_codes)

    send_dir_name = 'data/{}/test_jpg'.format(dataset)
    command_send_codes = 'scp -r %s/%s %s:%s/%s > /dev/null &' % ( 
        cur_dir,send_dir_name,host_list[idx], cur_dir,send_dir_name)

    print(command_send_codes)
    os.system(command_send_codes)

    
    
    




    






