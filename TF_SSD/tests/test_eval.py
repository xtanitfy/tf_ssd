import os
import numpy as np
from src.my_eval import Eval 

DETECTION_DIR = './tests/detections/data'
GT_DIR = './tests/labels'
CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                       'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                       'sofa', 'train', 'tvmonitor']
SCORE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5


    
if __name__ == '__main__':
    eval = Eval(CLASS_NAMES,GT_DIR,DETECTION_DIR,SCORE_THRESHOLD,IOU_THRESHOLD)
    eval.evaluate_det()
    
    
    
    
    
    
    
    
    