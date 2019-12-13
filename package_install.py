#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from common import *
import os

cur_dir = os.getcwd()

# start current node as name node
command=[]

for slave in host_list:
    if 'thumm01' not in slave:
        print('sending codes to slave [%s]' % slave)
        command_send_codes = 'ssh %s "mkdir -p %s ; rm -r %s/package"; scp -r package %s:%s/package > /dev/null' % (slave, cur_dir, cur_dir, slave, cur_dir)
        os.system(command_send_codes)
        command_send_codes = 'ssh %s "pip3 install --user %s/package/* -i https://pypi.tuna.tsinghua.edu.cn/simple" ; \
                              ssh %s "pip3 install --user matplotlib tqdm progress -i https://pypi.tuna.tsinghua.edu.cn/simple" &'%(slave,cur_dir,slave)
        os.system(command_send_codes)