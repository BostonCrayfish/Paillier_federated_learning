from models.Trainer import *
dataset='mnist'
class_num=10

net_glob,train_loder,test_loader=get_net_and_loader(dataset=dataset,mode="ALL")

traindata=train_loder.dataset.train_data
targetdata =train_loder.dataset.targets

import os
from PIL import Image
from tqdm import tqdm

def show(x,i):
    data=x.cpu().numpy()
    im = Image.fromarray(data)
    im.save(i)
    
## all data
dir_name='data/{}/all_train_jpg'.format(dataset)
    
for i in range(class_num):
    child_dir_name='{}/{}/'.format(dir_name,i)
    if not os.path.isdir(child_dir_name):
        mkdir_p(child_dir_name)
    
for i in tqdm(range(len(traindata))):
    #show(traindata[i],'{}/{}/{}_{}.jpg'.format(dir_name,targetdata[i],targetdata[i],i))
    # Have Done !
    pass

## part data
from models.sampling import *

class DatasetSplit(Dataset):
    def __init__(self, traindata,targetdata, idxs):
        self.traindata = traindata
        self.targetdata= targetdata
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        print(self.idxs[item])
        image, label = self.traindata[self.idxs[item]],self.targetdata[self.idxs[item]]
        return image, label
    
def save(traindata,targetdata,dir_name):
    if not os.path.isdir(dir_name):
        mkdir_p(dir_name)
    
    for i in range(class_num):
        child_dir_name='{}/{}/'.format(dir_name,i)
        if not os.path.isdir(child_dir_name):
            mkdir_p(child_dir_name)
    
    for i in tqdm(range(len(dataset))):
        data=traindata[i].cpu().numpy()
        im = Image.fromarray(data)
        path='{}/{}/{}_{}.jpg'.format(dir_name,targetdata[i],targetdata[i],i)
        im.save(path)

dict_users = mnist_iid(traindata, 4)

for idx in dict_users:
    x=DatasetSplit(traindata,targetdata, dict_users[idx])
    dir_name='data/{}/part_train_jpg/part_{}'.format(dataset,idx)
    #save(dataset.traindata,dataset.targetdata,dir_name)
    #Have done!

tdata=test_loader.dataset.test_data
ttarget =test_loader.dataset.targets

dir_name='data/{}/test_jpg'.format(dataset)
save(tdata,ttarget,dir_name)




