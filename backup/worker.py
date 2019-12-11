from models.Trainer import *

net,train_loder,test_loader=get_net_and_loader()
trainer=Trainer(net,train_loder,test_loader)

## To do : receive request and w
## For debug:
w=net.state_dict()

w=trainer.train(w)

print(w)
## To do 
## w-- (pallier) --> w1 -- (Aggregator) --> w_update -- (worker)--> w
