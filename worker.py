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

    def run(self):
        listen_worker = socket.socket()
        trainer=self.trainer
        try:
            # 监听端口
            listen_worker.bind(("0.0.0.0", worker_port[node_NO]))
            # For Debug
            #listen_worker.bind(("0.0.0.0", debug_port))
            listen_worker.listen(5)
            print("Listen started at port:%s"%(worker_port[node_NO]))

            while True:
                # 等待连接，连接后返回通信用的套接字
                sock_fd, addr = listen_worker.accept()
                print("connected by {}".format(addr))
                try:
                    data=recv_msg(sock_fd)
                    print("+++++++++++++++++++++")
                    w = pickle.loads(data)
                    w = trainer.train(w)
                    ## 序列化在send_data里面完成
                    self.send_data(w)
                except Exception as e:  # 如果出错则打印错误信息
                    traceback.print_exc()
                finally:
                    sock_fd.close()  # 释放连接
        except KeyboardInterrupt:  # 如果运行时按Ctrl+C则退出程序
            pass
        except Exception as e:  # 如果出错则打印错误信息
            traceback.print_exc()
        finally:
            listen_worker.close()  # 释放连接


    def send_data(self,data):
        request = "upload"
        num_try = 5
        
        print("send to {} {}".format(host_list[0], ag_port))
        while True and num_try>=0:
            try:
                num_try -= 1
                worker_sock = socket.socket()
                worker_sock.connect((host_list[0], ag_port))
                data_send=(request,data)
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