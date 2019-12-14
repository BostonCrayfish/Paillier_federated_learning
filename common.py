import socket

# 集群中的主机列表
all_host_list = ['thumm01','thumm02', 'thumm03','thumm04','thumm05', 'thumm06','thumm07']
all_worker_port = [12121, 11132, 11133,11134,11135,11136,11137]

n_nodes = 2
host_list = all_host_list[:n_nodes]
worker_port = all_worker_port[:n_nodes]

BUF_SIZE = 40960

ag_host = 'thumm01'
ag_port = 11111

#获取计算机名称
host_name=socket.gethostname()
node_NO = host_list.index(host_name) #本机序号，0~n_nodes-1

## 一个aggregator迭代的次数
glob_epochs = 20

host_ip_dict=dict({
    '192.168.0.101':"thumm01",
    '192.168.0.102':"thumm02",
    '192.168.0.103':"thumm03",
    '192.168.0.104':"thumm04",
    '192.168.0.105':"thumm05",
    '192.168.0.106':"thumm06",
    '192.168.0.107':"thumm07",
})

## 是否使用加密算法
is_paillier = False

cur_dir = '/mnt/data/dsjxtjc/2019211333/Paillier_federated_learning'