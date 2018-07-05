import os
import numpy as np
import shutil 

TEST_TXT = 'test.txt'
VOC_TEST_DIR = 'VOC2007TEST'
SRC_VOC_DIR='../VOC/VOCdevkit'

ANNO_DIR = os.path.join(VOC_TEST_DIR,'Annotations')
IMG_DIR = os.path.join(VOC_TEST_DIR,'JPEGImages')
TEST_DIR = os.path.join(VOC_TEST_DIR,'ImageSets/Main')
DST_TEST_TXT = os.path.join(TEST_DIR,'test.txt')

if os.path.exists(ANNO_DIR) == False:
    os.makedirs(ANNO_DIR)
    
if os.path.exists(IMG_DIR) == False:
    os.makedirs(IMG_DIR)

if os.path.exists(TEST_DIR) == False:
    os.makedirs(TEST_DIR)
    
if os.path.exists(DST_TEST_TXT) == False:
    os.mknod(DST_TEST_TXT)
    

with open(TEST_TXT,'r') as f:
    lines = f.readlines()

with open(DST_TEST_TXT,'w+') as f:
    for line in lines:
        items = line.strip().split(' ')
        imgfile = items[0] 
        annofile = items[1] 
        #shutil.copy(imgfile,IMG_DIR)
        shutil.copy(os.path.join(SRC_VOC_DIR,annofile),ANNO_DIR)
        
        name = annofile.split('/')[-1].split('.')[0]
        f.write(name+'\n')

