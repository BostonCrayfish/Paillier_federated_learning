from models.Trainer import *

net,train_loder,test_loader=get_net_and_loader()
trainer=Trainer(net,train_loder,test_loader)
w=net.state_dict()
print(w)
## To do 
## w-- (pallier) --> w1 -- (Aggregator) --> w_update -- (worker)--> w


w=trainer.train(w)
w=trainer.train(w)