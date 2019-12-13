#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from common import *
import os

cur_dir = os.getcwd()

import sys


# start current node as name node

for slave in host_list:
    if 'thumm01' not in slave:
        print('sending codes to slave [%s]' % slave)
        if sys.argv[1]=='all':
            command_send_codes = 'ssh %s "mkdir -p %s"; scp -r ./* %s:%s > /dev/null' % (slave, cur_dir, slave, cur_dir)
        elif sys.argv[1]=='code':
            command_send_codes = 'ssh %s "mkdir -p %s"; scp -r *.py %s:%s > /dev/null' % (slave, cur_dir, slave, cur_dir)
        os.system(command_send_codes)
    # command_start = 'ssh %s "python3 %s/data_node.py " &' % (slave, cur_dir)
    # print('trying to start data node [%s]' % slave)
    # os.system(command_start)