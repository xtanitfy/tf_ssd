import tensorflow as tf
import numpy as np
import time
import sys
import os
from struct import *

def io_save_array_to_txt_file(filename,arr):
    act_flat = arr.reshape(arr.size)
    
    print ('save ',filename)
    filename = filename.split('.')[-2] + '.txt'
    file = open(filename, "w")
    cnt = 0
    for i in range(0,arr.size):
        str = '%-12f' % act_flat[i]
        file.write(str)
        if (cnt % 64 == 0 and cnt != 0):
            file.write('\n')
        cnt = cnt + 1
    file.close()
    
def io_save_array_to_bin_file(filename,data):
    
    filename = filename.split('.')[-2] + '.bin'
    file = open(filename, "wb")
    
    act_flat = data.reshape(data.size)
    
    for i in range(0,act_flat.shape[0]):
        file.write(pack("f",float(act_flat[i])))
    file.close()
    
def io_save_list3_to_bin_file(filename,data):
    a = []
    for i in range(len(data)):
        print (str(len(data[i]))+' ')
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                a.append(data[i][j][k])
    
    io_save_array_to_bin_file(filename,np.array(a))
 
def io_save_list3_to_txt_file(filename,data):
    a = []
    for i in range(len(data)):
        print (str(len(data[i]))+' ')
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                a.append(data[i][j][k])
    
    io_save_array_to_txt_file(filename,np.array(a))

    
def io_read_from_bin_file(dir,filename,size):
    arr = np.zeros((int(size)),dtype='float32')
    filefullname = os.path.jopin(dir,filename)

    is_exist = os.path.isfile(filefullname)
    assert is_exist == True
    
    file = open(filefullname, "rb")
    #file_size = os.path.getsize(filefullname)

    for i in range(0,int(size)):
        data = unpack("f",file.read(4))
        arr[i] = data[0]
    file.close()
    
    return arr
    

 