from common import *
import os
import time 

cur_dir = os.getcwd()

import argparse
parser = argparse.ArgumentParser(description='Worker')

parser.add_argument('--mode',default='distributed',type=str)

parser.add_argument('--dataset', default='mnist', type=str)
parser.add_argument('--model', default='mlp', type=str)
parser.add_argument('--itype', default='iid', type=str)
parser.add_argument('--epoch',default=100,type=int)
parser.add_argument('--l_batch',default=400,type=int)

parser.add_argument('--path',default='400',type=int)
parser.add_argument('--part',default=4,type=int)



args = parser.parse_args()

# start workers
for slave in host_list:
    command_start = 'ssh %s "python3 %s/worker.py --dataset %s --model %s --l_batch %s " &' % (slave, cur_dir,args.dataset,args.model,args.l_batch)
    print('trying to start worker [%s]' % slave)
    os.system(command_start)

time.sleep(5)
# start aggregator
command_start = 'python3 %s/aggregator.py --dataset %s --model %s --epoch %s' % (cur_dir,args.dataset,args.model,args.epoch)
# #print('command to start name node: ', command_start)
# #TO do： 开启远程的aggregator
print('trying to start aggregator')
os.system(command_start)
