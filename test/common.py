import socket

# 集群中的主机列表
host_list = ['thumm01','thumm02', 'thumm03', 'thumm04']
BUF_SIZE = 2000000

n_nodes = 2
ag_port = 33321
worker_port = [11017, 11018, 11019,11020,11021]

#获取计算机名称
host_name=socket.gethostname()
#host_name = 'localhost'
node_NO = host_list.index(host_name) #本机序号，0~n_nodes-1
