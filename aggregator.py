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
from models.logger import *

import pickle
import traceback

from phe import paillier
from functools import reduce 
import time 



class Aggregator(object):
    def __init__(self,glob_epochs=200,dataset='mnist',net_glob='mlp',itype='iid'):
        self.net_glob,self.train_loder,self.test_loader=get_net_and_loader(model_name=net_glob,dataset=dataset)
        self.glob_w=self.net_glob.state_dict()
        self.glob_epochs=glob_epochs
        dir_name = 'checkpoint/aggregator/{}-{}-{}{}p'.format(dataset,net_glob,itype,n_nodes)
        if not os.path.isdir(dir_name):
            mkdir_p(dir_name)
        self.logger = Logger(os.path.join(dir_name, 'log.txt'), title='{}-{}'.format(dataset,net_glob))
        self.logger.set_names(['test Acc','test Loss','time'])
        self.time_start=time.time()


    ## To do ：把collect_answer直接写成run
    def run(self):  
        print("Aggregator started")
        # Debug: 先计算一次精度，看看数据集情况
        #compute_acc(net_glob,self.test_loader)
        ## 分发参数
        socket_list=self.send_new_data2(self.glob_w)
        ## 同步SGD
        self.collect_answer2(socket_list)
       

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
                    #sock_fd.setblocking(0)
                    print("connected by {}".format(addr))
                    # 获取请求方发送的指令
                    request=self.ag_receive(sock_fd)
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
                            #self.glob_w=FedAvg(w_locals)
                            self.send_new_data1(self.glob_w)
                            w_locals=[]
                            self.net_glob.load_state_dict(self.glob_w)
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
                finally:
                    sock_fd.close()  # 释放连接         
        except KeyboardInterrupt:  # 如果运行时按Ctrl+C则退出程序
            pass
        except Exception as e:  # 如果出错则打印错误信息
            traceback.print_exc()
        finally:
            listen_ag.close()  # 释放连接


 
    def collect_answer2(self,socket_list):
        w_locals=[]
        encrypt_random_num_locals=[]
        net_glob,train_loder,test_loader=self.net_glob,self.train_loder,self.test_loader
        glob_w=self.glob_w
        epoch = 1
        from tqdm import tqdm
        while True:
            try:
                ### 放在while里面的逻辑太复杂了，debug都是泪
                for data_node_sock in socket_list:
                    request=self.ag_receive(data_node_sock)
                    ip=data_node_sock.getpeername()[0]
                    print('received result from %s'%(host_ip_dict[ip]))
                    # For Debug：
                    # response_msg = pickle.dumps(glob_w)
                    # request=pickle.loads(response_msg)
                    if request[0]=='upload':
                        w_receive=request[1]
                        w_locals.append(w_receive)
                        if is_paillier==True:
                           encrypt_random_num_locals.append(request[2])     

                    data_node_sock.close()
                    socket_list.remove(data_node_sock)

                ### 应该写一个初始化的函数，直接调用
                if len(socket_list)==0:
                    self.glob_w=FedAvg(w_locals)
                    #### 实际上aggregator没有密钥，这里是为了查看实验结果
                    if is_paillier==True:
                        f = open('%s/key/private_key' % (cur_dir), 'rb')
                        private_key = pickle.load(f)
                        ## 随机数密钥求和
                        encrypt_random_sum=reduce(lambda x, y: x + y, encrypt_random_num_locals)
                        ## 和密码解密/结点数目 = 随机数平均值
                        random_avg = private_key.decrypt(encrypt_random_sum)/len(encrypt_random_num_locals)
                        print(random_avg)
                        ## 参数平均值减去密钥平均值
                        self.glob_w = dict({key: (value - random_avg) for key,value in self.glob_w.items()})

                    print("Epoch:{}/{} Received all parm and update, Test:".format(epoch,self.glob_epochs)) 
                    epoch +=1
                    self.net_glob.load_state_dict(self.glob_w)
                    end_time = time.time()
                    compute_acc(self.net_glob,self.test_loader,self.logger,end_time-self.time_start)
                    if epoch == self.glob_epochs:
                        break
                    ## next stage
                    socket_list=self.send_new_data2(self.glob_w) 
                    
            except KeyboardInterrupt:
                exit()
            except Exception:
                traceback.print_exc()

    ### 收到空数据会出错,包括了反序列化的过程
    ### Todo: 构成成对的函数，反序列化写在里面，在worker里构建对应的加密和序列化函数
    def ag_receive(self, sock_fd):
        data=recv_msg(sock_fd)
        data = pickle.loads(data)
        #print("Data received")
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
        print("Send parms to {}".format(list((host_list[i], worker_port[i]) for i in range(n_nodes))))


    def send_new_data2(self,data_new):
        socket_list=[]
        data_new=pickle.dumps(dict(data_new))
        num_try=5
        for i in range(n_nodes):
            while num_try>=0:
                try:
                    num_try -=1
                    renew_sock = socket.socket()
                    #print(host_list[i], worker_port[i])
                    renew_sock.connect((host_list[i], worker_port[i]))
                    send_msg(renew_sock,data_new)
                    socket_list.append(renew_sock)
                    time.sleep(0.1)
                    break   
                except Exception:
                    traceback.print_exc() 
        print("Send the parm to {}".format(list((host_list[i], worker_port[i]) for i in range(n_nodes))))
        if len(socket_list)==0:
            print("No workers!")
            exit()
        return socket_list




if __name__ == '__main__':
    import os
    
    import argparse
    parser = argparse.ArgumentParser(description='Worker')
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--model', default='mlp', type=str)
    parser.add_argument('--itype', default='iid', type=str)
    parser.add_argument('--epoch',default=50,type=int)
    args = parser.parse_args()

    aggr = Aggregator(glob_epochs=args.epoch,dataset=args.dataset,net_glob=args.model,itype=args.itype)
    aggr.run()
