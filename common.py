import socket

# 集群中的主机列表
host_list = ['thumm01','thumm02', 'thumm03', 'thumm04']
BUF_SIZE = 40960

n_nodes = 4
ag_host = 'thumm01'
ag_port = 22222
worker_port = [11771, 11772, 11773,11774,11775]

#获取计算机名称
host_name=socket.gethostname()
#host_name = 'localhost'
node_NO = host_list.index(host_name) #本机序号，0~n_nodes-1

## 一个aggregator迭代的次数
glob_epochs = 20