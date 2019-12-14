from common import *
import os
import time 

cur_dir = os.getcwd()


# start worker
for slave in host_list:
    command_start = 'ssh %s "python3 %s/worker.py " &' % (slave, cur_dir)
    print('trying to start worker [%s]' % slave)
    os.system(command_start)


time.sleep(5)
# start aggregator
command_start = 'python3 %s/aggregator.py' % cur_dir
# #print('command to start name node: ', command_start)
# #TO do： 开启远程的aggregator
print('trying to start aggregator')
os.system(command_start)
