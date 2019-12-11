import math
import os
import socket
import time
import numpy as np
import pandas as pd

from common import *

import numpy as np 
from models.Trainer import *
from models.Federated import *
from models.utils import *

import pickle
import traceback

## 一个aggregator迭代的次数
glob_epochs = 4
num_workers = 1


class Aggregator(object):
    def __init__(self):
        self.net_glob,self.train_loder,self.test_loader=get_net_and_loader()
        self.glob_w=self.net_glob.state_dict()

    
    def run(self):  
        print("Aggregator started")
        net_glob,train_loder,test_loader=self.net_glob,self.train_loder,self.test_loader
        glob_w=self.glob_w
        #compute_acc(net_glob,train_loder)

        for iter in range(glob_epochs):
            print("Epoch:{}".format(iter))
            w_locals = []
            idxs_users = range(num_workers)
            #socket_list=[]
            
            from tqdm import tqdm
            ## 分发参数
            print('Map Step:')
            self.send_new_data(pickle.dumps(glob_w))

            # 测试用 ##############
            # for i in tqdm(idxs_users):
            #     renew_sock = socket.socket()
            #     renew_sock.connect((host_list[i], worker_port[i]))
            #     renew_sock.send(pickle.dumps(glob_w))
            #     time.sleep(0.2)
            #     socket_list.append((str(i),renew_sock))
        
            ## 同步SGD
            w_locals=self.collect_answer1()
            #w_locals=self.collect_answer2(socket_list)
            

            w_glob=FedAvg(w_locals)
            net_glob.load_state_dict(w_glob)
            compute_acc(net_glob,test_loader)

    def collect_answer1(self):
        listen_ag = socket.socket()
        upload_times = 0 #初始化收到参数的次数
        data = []   #初始化接收到的数据

        try:
            # 监听端口
            listen_ag.bind(("0.0.0.0", ag_port))
            listen_ag.listen(5)
            while True:
                # 等待连接，连接后返回通信用的套接字
                sock_fd, addr = listen_ag.accept()
                print("connected by {}".format(addr))
                try:
                    # 获取请求方发送的指令
                    request = str(sock_fd.recv(128), encoding='utf-8')
                    request = request.split()  # 指令之间使用空白符分割
                    print("Request: {}".format(request))

                    cmd = request[0]  # 指令第一个为指令类型

                    if cmd == "upload":
                        data.append(self.upload(sock_fd))
                        upload_times += 1
                        response = "Upload succeeds"
                        if upload_times == n_nodes:
                            #data_new = self.mean(data) #mean函数将处理n_nodes个csv数据
                            upload_times = 0
                            self.send_new_data(data_new)
                    else:  # 其他位置指令
                        response = "Undefined command: " + " ".join(request)

                    print("Response: {}".format(response))
                    sock_fd.send(bytes(response, encoding='utf-8'))
                except KeyboardInterrupt:  # 如果运行时按Ctrl+C则退出程序
                    break
                except Exception as e:  # 如果出错则打印错误信息
                    print(e)
                finally:
                    sock_fd.close()  # 释放连接
        except KeyboardInterrupt:  # 如果运行时按Ctrl+C则退出程序
            pass
        except Exception as e:  # 如果出错则打印错误信息
            print(e)
        finally:
            listen_ag.close()  # 释放连接


    ## 用于同步SGD，建立一一对应关系，debug用
    def collect_answer2(self,socket_list):
        w_locals=[]
        net_glob,train_loder,test_loader=self.net_glob,self.train_loder,self.test_loader
        glob_w=self.glob_w
        from tqdm import tqdm
        print('Reduce Step:')
        while True:
            try:
                for data in socket_list:
                    data_node_sock=data[1]
                    #response_msg = recv_msg(data_node_sock)
                    # For Debug：
                    response_msg = pickle.dumps(glob_w)
                    w_receive=pickle.loads(response_msg)
                    ####################################
                    w_locals.append(w_receive)
                    data_node_sock.close()
                    socket_list.remove(data)
                # ## To do : 其它停止条件，如超时
                if len(socket_list)==0:
                    print("received all parm")
                    return w_locals
            except KeyboardInterrupt:
                break
            except Exception:
                traceback.print_exc()

    

    ### 收到空数据会出错
    def upload(self, sock_fd):
        data = sock_fd.recv(BUF_SIZE)
        data = pickle.load(data_new)
        print("Data received")
        return data

    def mean(self,data):
        #聚合函数，求均值
        #data是一个列表，里面有n_nodes个csv
        return data[0]+data[1]+data[2]

    def send_new_data(self,data_new):
        #data_new = str(data_new)
        # data_new is a model state dict
        data_new=pickle.dumps(data_new)
        for i in range(n_nodes):
            renew_sock = socket.socket()
            renew_sock.connect((host_list[i], worker_port[i]))
            #renew_sock.send(bytes(data_new, encoding='utf-8'))
            renew_sock.send(data_new)
            renew_sock.close()
            time.sleep(0.2)

    




aggr = Aggregator()
aggr.run()
