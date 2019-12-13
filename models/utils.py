from common import *
import struct
import time
import pickle
import traceback

# Use a header to indicate data size, refer to https://stackoverflow.com/a/27429611
def send_msg(sock, data):
    # Pack the data size into an int with big-endian
    header = struct.pack('>i', len(data))
    sock.sendall(header)
    try:
        sock.sendall(data)
    except:
        traceback.print_exc()
    
    


def recv_msg(sock):
    # Parse the header, which is a 4-byte int
    header = sock.recv(4)
    num_try = 5
    while len(header) == 0 and num_try:
        print('Fail to receive data. Trying again...')
        time.sleep(0.1)
        header = sock.recv(4)
        num_try -= 1
    data = bytearray()
    if len(header) != 0:
        data_size = struct.unpack('>i', header)[0]
        while len(data) < data_size:
            part = sock.recv(BUF_SIZE)
            data.extend(part)

    #print(pickle.loads(data))
    #print("what's more!",sock.recv(BUF_SIZE))      
    return data