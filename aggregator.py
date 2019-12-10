import numpy as np 
from models.Trainer import *
from models.Federated import *

glob_epochs = 10
num_workers = 4

net_glob,train_loder,test_loader=get_net_and_loader()

for iter in range(glob_epochs):
    w_locals = []
    idxs_users = range(num_workers)


    ## To do :
    ## sock=send_request(idxs_users)
    ## w_locals=receive_weight(sock)
    ##
    
    w_locals.append(net_glob.state_dict())
    w_locals.append(net_glob.state_dict())

    w_glob=FedAvg(w_locals)
    

    net_glob.load_state_dict(w_glob)
    compute_acc(net_glob,test_loader)

