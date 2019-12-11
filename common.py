import socket

# 集群中的主机列表
host_list = ['localhost','thumm02', 'thumm03', 'thumm04','198']
BUF_SIZE = 2000000

n_nodes = 3
ag_port = 21017
worker_port = [22, 11018, 11019]

#获取计算机名称
host_name=socket.gethostname()
node_NO = host_list.index(host_name) #本机序号，0~n_nodes-1