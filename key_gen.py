from phe import paillier
import pickle
import os
import os
from common import *

cur_dir = os.getcwd()

## 模拟第三方产生密钥

def mkdir_p(path):
    '''make dir if not exist'''
    import os
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


public_key,private_key=paillier.generate_paillier_keypair()


child_dir_name="key"
if not os.path.isdir(child_dir_name):
    mkdir_p(child_dir_name)

f = open('%s/public_key'% (child_dir_name), 'wb')
pickle.dump(public_key, f)

f = open('%s/private_key'% (child_dir_name), 'wb')
pickle.dump(private_key, f)


for slave in host_list:
    if 'thumm01' not in slave:
        print('sending codes to slave [%s]' % slave)
        command_send_codes = 'ssh %s "mkdir -p %s/key"; scp -r key/* %s:%s/key/ > /dev/null' % (slave, cur_dir, slave, cur_dir)
        os.system(command_send_codes)
