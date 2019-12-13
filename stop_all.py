from common import *
import os

cur_dir = os.getcwd()
command_kill_pattern = 'kill $(lsof -t -i:%d)'

# stop name node
command_kill = command_kill_pattern % ag_port
print('trying to kill aggregator')
os.system(command_kill)

# stop all data nodes
command_kill = command_kill_pattern
for i in range(len(host_list)):
    command_remote_kill = 'ssh %s \'%s\'' % (host_list[i], command_kill % worker_port[i])
    print('trying to kill data node [%s]' % host_list[i])
    os.system(command_remote_kill)