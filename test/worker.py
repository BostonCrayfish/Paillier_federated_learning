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
            print("Listen started")

            while True:
                # 等待连接，连接后返回通信用的套接字
                sock_fd, addr = listen_worker.accept()
                print("connected by {}".format(addr))

                try:
                    data=recv_msg(sock_fd)
                    w = pickle.loads(data)
                    w = trainer.train(w)
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
        while True:
            try:
                worker_sock = socket.socket()
                print("send to {} {}".format(host_list[0], ag_port))
                worker_sock.connect((host_list[0], ag_port))
                ## Debug：
                #worker_sock.connect((host_list[0], ag_port))
                #worker_sock.send(bytes(request, encoding='utf-8'))
                #send_msg(worker_sock,request)
                #time.sleep(0.02)
                data_send=(request,data)
                data_send=pickle.dumps(data_send)
                send_msg(worker_sock,data_send)
                worker_sock.close()
                break
            except:
                traceback.print_exc()

        return "Send data successfully"

worker=Worker()
worker.run()