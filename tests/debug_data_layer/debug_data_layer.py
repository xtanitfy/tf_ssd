import tensorflow as tf
import numpy as np
import os
import os.path as osp
import subprocess
import fileinput
from os.path import join, getsize  
from struct import *
import math
import numpy as np
import cv2
from config import config
from data_layer import Data_layer
import fileinput

TEST_IMAGE_FILE = 'data_source/debug_2.jpg'
TEST_IMAGE_ANNO_FILE = 'data_source/debug_2_anno.txt'


def load_anno(filename):
    annos = []
    for line in fileinput.input(filename):
        items = line.strip().split(' ')
        items_float = [float(items[i]) for i in range(0,len(items))]
        annos.append(items_float)
    return annos

if __name__ == '__main__':
    mc = config()
    ori_image = cv2.imread(TEST_IMAGE_FILE)
    ori_height, ori_width, ori_channel = [float(v) for v in ori_image.shape]
    
    annos = load_anno(TEST_IMAGE_ANNO_FILE)
    annos = np.array(annos)
    gt_boxes = annos[:,2:]
    
    print gt_boxes
    
    data_layer = Data_layer(mc)
    image,gt_boxes,_ = data_layer.Process(ori_image,gt_boxes)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    