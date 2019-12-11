from models.Trainer import *
import os

for i in range(10):
    dir_name='data/{}/'.format(i)
    if not os.path.isdir(dir_name):
        mkdir_p(dir_name)