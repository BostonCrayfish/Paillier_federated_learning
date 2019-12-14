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


class Worker(object):
    def __init__(self):
        self.net_glob,self.train_loder,self.test_loader=get_net_and_loader()
        self.trainer=Trainer(self.net_glob,self.train_loder,self.test_loader)
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
                    print(w)
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


worker=Worker()
worker.run()