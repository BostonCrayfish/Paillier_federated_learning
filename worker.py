# -*- coding: utf-8 -*-
# Python version: 3.6


import math
import os
import socket
import time
import numpy as np
import pandas as pd

from common import *
from models.Trainer import *
import pickle
import traceback
from models.utils import *

#获取计算机名称
host_name=socket.gethostname()
node_NO = host_list.index(host_name) #本机序号，0~n_nodes-1

class Worker(object):
    def __init__(self,dataset='mnist',net_glob='mlp',itype='iid',batch_each_epoch=100):
        self.net_glob,self.train_loder,self.test_loader=get_net_and_loader(model_name=net_glob,dataset=dataset)
        self.trainer=Trainer(self.net_glob,self.train_loder,self.test_loader,batch_each_epoch=batch_each_epoch)
        if is_paillier==True:
            self.random_num=np.random.randn(1)

    def run(self):
        listen_worker = socket.socket()
        trainer=self.trainer
        try:
            # 监听端口
            print("Listen started at port:%s"%(worker_port[node_NO]))
            listen_worker.bind(("0.0.0.0", worker_port[node_NO]))
            # For Debug
            #listen_worker.bind(("0.0.0.0", debug_port))
            listen_worker.listen(5)

            while True:
                # 等待连接，连接后返回通信用的套接字
                sock_fd, addr = listen_worker.accept()
                print("connected by {}".format(addr))
                try:
                    data=recv_msg(sock_fd)
                    w = pickle.loads(data)
                    w = trainer.train(w)
                    ## 序列化在send_data里面完成
                    self.send_data2(sock_fd,w)
                except Exception as e:  # 如果出错则打印错误信息
                    traceback.print_exc()
                finally:
                    sock_fd.close()  # 释放连接
        except KeyboardInterrupt:  # 如果运行时按Ctrl+C则退出程序
            listen_worker.close()  # 释放连接
        except Exception as e:  # 如果出错则打印错误信息
            traceback.print_exc()
        finally:
            listen_worker.close()  # 释放连接

    

    def send_data1(self,data):
        request = "upload"
        num_try = 5
        
        print("send to {} {}".format(ag_host, ag_port))
        while True and num_try>=0:
            try:
                num_try -= 1
                worker_sock = socket.socket()
                worker_sock.connect((ag_host, ag_port))
                data_send=(request,data)
                data_send=pickle.dumps(data_send)
                send_msg(worker_sock,data_send)
                worker_sock.close()
                time.sleep(0.2)
                break
            except:
                traceback.print_exc()
        print("Send data successfully")


    def send_data2(self,sock,data):
        request = "upload"
        num_try = 5
        print("send to {} {}".format(ag_host, ag_port))
        while True and num_try>=0:
            try:
                num_try -= 1
                worker_sock=sock
                data_send=(request,data) 
                if is_paillier==True:
                    ## 参数加上随机数
                    data = dict({key:value+torch.from_numpy(self.random_num).float() for key,value in data.items()})
                    f = open('%s/key/public_key' % (cur_dir), 'rb')
                    public_key = pickle.load(f)
                    print(self.random_num[0])
                    encrypt_random_num=public_key.encrypt(self.random_num[0].astype(np.float64))
                    data_send=(request,data,encrypt_random_num)

                data_send=pickle.dumps(data_send)
                send_msg(worker_sock,data_send)
                worker_sock.close()
                time.sleep(0.2)
                break
            except:
                traceback.print_exc()
        print("Send data successfully")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Worker')
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--model', default='mlp', type=str)
    parser.add_argument('--itype', default='iid', type=str)
    parser.add_argument('--l_batch',default=400,type=int)
    args = parser.parse_args()

    worker=Worker(dataset=args.dataset,net_glob=args.model,itype=args.itype,batch_each_epoch=args.l_batch)
    worker.run()