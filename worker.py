import math
import os
import socket
import time
import numpy as np
import pandas as pd

from common import *



class Trainer:
    def train(self):
        x=np.random.rand()
        return "This is a test: " + str(x)
    def listen(self):
        listen_worker = socket.socket()

        try:
            # 监听端口

            listen_worker.bind(("0.0.0.0", worker_port[node_NO]))
            listen_worker.listen(5)
            print("Listen started")

                # 等待连接，连接后返回通信用的套接字
            sock_fd, addr = listen_worker.accept()
            print("connected by {}".format(addr))

            try:
                data = sock_fd.recv(BUF_SIZE)
                print("data received")
            except Exception as e:  # 如果出错则打印错误信息
                print(e)
            finally:
                sock_fd.close()  # 释放连接
        except KeyboardInterrupt:  # 如果运行时按Ctrl+C则退出程序
            pass
        except Exception as e:  # 如果出错则打印错误信息
            print(e)
        finally:
            listen_worker.close()  # 释放连接

        return data

    def send_data(self,data):
        worker_sock = socket.socket()
        request = "upload"
        while True:
            try:
                worker_sock.connect((host_list[0], ag_port))
                worker_sock.send(bytes(request, encoding='utf-8'))
                time.sleep(0.02)
                worker_sock.send(bytes(data, encoding='utf-8'))
                worker_sock.close()
                break
            except:
                time.sleep(0.02)

        return "Send data successfully"

trainer = Trainer()
data = trainer.train()
print(data)
trainer.send_data(data)
data = trainer.listen() #将接收到更新后的数据
print(data)