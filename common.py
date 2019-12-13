import socket

# 集群中的主机列表
host_list = ['thumm01','thumm02', 'thumm03', 'thumm04']
BUF_SIZE = 40960

n_nodes = 4
ag_host = 'thumm01'
ag_port = 22222
worker_port = [10111, 10112, 10113,10114]

#获取计算机名称
host_name=socket.gethostname()
#host_name = 'localhost'
node_NO = host_list.index(host_name) #本机序号，0~n_nodes-1

## 一个aggregator迭代的次数
glob_epochs = 20

host_ip_dict=dict({
    '192.168.0.101':"thumm01",
    '192.168.0.102':"thumm02",
    '192.168.0.103':"thumm03",
    '192.168.0.104':"thumm04",
})