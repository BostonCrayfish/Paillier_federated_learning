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
glob_epochs = 20


class Aggregator(object):
    def __init__(self):
        self.net_glob,self.train_loder,self.test_loader=get_net_and_loader()
        self.glob_w=self.net_glob.state_dict()
        self.glob_epochs=glob_epochs

    
    def run(self):  
        print("Aggregator started")
        net_glob,train_loader,test_loader=self.net_glob,self.train_loder,self.test_loader
        glob_w=self.glob_w
        # Debug:
        #compute_acc(net_glob,test_loader)

        ## 分发参数
        self.send_new_data1(glob_w)
        ## 同步SGD
        self.collect_answer1()
       

    def collect_answer1(self):
        listen_ag = socket.socket()
        listen_ag.setblocking(0)
        upload_times = 0 #初始化收到参数的次数
        w_locals = []   #初始化接收到的数据
        try:
            # 监听端口
            listen_ag.bind(("0.0.0.0", ag_port))
            print("Listen on port {}".format(ag_port))
            listen_ag.listen(5)
            while True:
                try:
                    # 等待连接，连接后返回通信用的套接字
                    sock_fd, addr = listen_ag.accept()
                    sock_fd.setblocking(0)
                    print("connected by {}".format(addr))
                    # 获取请求方发送的指令
                    request=self.upload(sock_fd)
                    #request = str(sock_fd.recv(128), encoding='utf-8')
                    #request = request.split()  # 指令之间使用空白符分割
                    #print("Request: {}".format(request))
                    cmd = request[0]  # 指令第一个为指令类型
                    if cmd == "upload":
                        ## 接收数据
                        w_new=request[1]
                        w_locals.append(w_new)
                        
                        upload_times += 1
                        response = "Upload succeeds:%s"%(len(w_locals))

                        ### 这个判定条件只要有一次错误就会全部GG
                        if upload_times == n_nodes:
                            self.w_glob=FedAvg(w_locals)
                            self.send_new_data1(self.glob_w)
                            w_locals=[]
                            self.net_glob.load_state_dict(self.w_glob)
                            compute_acc(self.net_glob,self.test_loader)
                            upload_times = 0
                            self.glob_epochs-=1
                            if self.glob_epochs==0:
                                return 
                            
                    else:  # 其他位置指令
                        response = "Undefined command: " + " ".join(request)

                    print("Response: {}".format(response))
                    #sock_fd.send(bytes(response, encoding='utf-8'))

                except KeyboardInterrupt:  # 如果运行时按Ctrl+C则退出程序
                    sock_fd.close()  # 释放连接
                    break
                except BlockingIOError as e:
                    pass
                except Exception as e:  # 如果出错则打印错误信息
                    traceback.print_exc()
                # finally:
                #     sock_fd.close()  # 释放连接         
        except KeyboardInterrupt:  # 如果运行时按Ctrl+C则退出程序
            pass
        except Exception as e:  # 如果出错则打印错误信息
            traceback.print_exc()
        finally:
            listen_ag.close()  # 释放连接


 
    def collect_answer2(self,socket_list):
        w_locals=[]
        net_glob,train_loder,test_loader=self.net_glob,self.train_loder,self.test_loader
        glob_w=self.glob_w
        from tqdm import tqdm
        while True:
            try:
                for data_node_sock in socket_list:
                    response_msg = recv_msg(data_node_sock)
                    # For Debug：
                    # response_msg = pickle.dumps(glob_w)
                    w_receive=pickle.loads(response_msg)
                    ####################################
                    w_locals.append(w_receive)
                    data_node_sock.close()
                    socket_list.remove(data)
                # ## To do : 其它停止条件，如超时
                if len(socket_list)==0:
                    self.w_glob=FedAvg(w_locals)
                    self.net_glob.load_state_dict(self.w_glob)
                    compute_acc(self.net_glob,self.test_loader)
                    socket_list=self.send_new_data2(self.glob_w)
                    print("received all parm and update") 
            except KeyboardInterrupt:
                break
            except Exception:
                traceback.print_exc()

    ### 收到空数据会出错
    def upload(self, sock_fd):
        data=recv_msg(sock_fd)
        #data = sock_fd.recv(BUF_SIZE)
        data = pickle.loads(data)
        print("Data received")
        return data

    def send_new_data1(self,data_new):
        # data_new is a model state dict
        #print(data_new)
        data_new=pickle.dumps(dict(data_new))
        for i in range(n_nodes):
            ### important ! Or else it would be try for unlimited times when some worker was died !
            num_try=5
            while num_try>=0:
                try:
                    num_try -=1
                    renew_sock = socket.socket()
                    print(host_list[i], worker_port[i])
                    renew_sock.connect((host_list[i], worker_port[i]))
                    send_msg(renew_sock,data_new)
                    renew_sock.close()
                    time.sleep(0.2)
                    break   
                except Exception:
                    traceback.print_exc() 
        print("Send the parm to {}".format(list((host_list[i], worker_port[i]) for i in range(n_nodes))))


    def send_new_data2(self,data_new):
        socket_list=[]
        data_new=pickle.dumps(dict(data_new))
        num_try=5
        for i in range(n_nodes):
            while num_try>=0:
                try:
                    num_try -=1
                    renew_sock = socket.socket()
                    print(host_list[i], worker_port[i])
                    renew_sock.connect((host_list[i], worker_port[i]))
                    send_msg(renew_sock,data_new)
                    socket_list.append(renew_sock)
                    time.sleep(0.1)
                    break   
                except Exception:
                    traceback.print_exc() 
        print("Send the parm to {}".format(list((host_list[i], worker_port[i]) for i in range(n_nodes))))
        return socket_list





aggr = Aggregator()
aggr.run()
