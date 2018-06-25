import tensorflow as tf
import numpy as np
import os
import os.path as osp
import subprocess
from easydict import EasyDict as edict
import fileinput
from os.path import join, getsize  
from struct import *
import math
from mutibox_loss_layer import MutiBoxLossLayer


BATCH_SIZE   = 2
NUM_PRIORBOX = 8732
NUM_CLASSES  = 21
NUM_GT       = 6      

def config():
    cfg = edict()
    cfg.BATCH_SIZE = BATCH_SIZE
    cfg.NUM_PRIORBOX = NUM_PRIORBOX
    cfg.NUM_CLASSES = NUM_CLASSES
    cfg.NUM_GT = NUM_GT
    cfg.overlap_threshold = 0.5
    cfg.neg_pos_ratio = 3.
    cfg.neg_overlap = 0.5
    cfg.background_label_id = 0
    cfg.EPSILON = 1e-16 
    return cfg

def read_from_binfile(filename,size):
    arr = np.zeros((int(size)),dtype='float32')
    filename = "data/" + filename
    file = open(filename, "rb")
    file_size = os.path.getsize(filename) 
    assert size * 4 == file_size
    for i in range(0,int(size)):
        data = unpack("f",file.read(4))
        arr[i] = data[0]
    file.close()
    return arr

def save_array_to_txt_file(filename,arr): 
    arr_flat = arr.reshape(arr.size)
    filename = "out/" + filename
    print ('save ',filename)
    file = open(filename, "w")
    cnt = 0
    for i in range(0,arr.size):
        str = '%-12f' % arr_flat[i]
        file.write(str)
        if (cnt % 64 == 0 and cnt != 0):
            file.write('\n')
        cnt = cnt + 1
    file.close()
    
def load_data(mc):
    size = BATCH_SIZE * NUM_PRIORBOX * 4
    loc_data = read_from_binfile("loc_data.bin",size)
    loc_data = np.reshape(loc_data,(BATCH_SIZE,NUM_PRIORBOX,4))
    
    size = BATCH_SIZE * NUM_PRIORBOX * NUM_CLASSES
    conf_data = read_from_binfile("conf_data.bin",size)
    conf_data = np.reshape(conf_data,(BATCH_SIZE,NUM_PRIORBOX,NUM_CLASSES))

    size = 2 * NUM_PRIORBOX * 4
    prior_data = read_from_binfile("prior_data.bin",size)
    prior_data = np.reshape(prior_data,(2,NUM_PRIORBOX,4))

    size = NUM_GT * 8
    gt_data = read_from_binfile("gt_data.bin",size)
    gt_data = np.reshape(gt_data,(NUM_GT,8))

    return loc_data,conf_data,prior_data,gt_data

def save_debug_var(sess,feed_dict,debug_val,debug_val_names):

    for i in range(0,len(debug_val)):
        var = sess.run(debug_val[i],feed_dict=feed_dict)
        print ("save " + debug_val_names[i] + ' shape:' + str(var.shape))
        
        filename = '{}.txt'.format(debug_val_names[i]) 
        save_array_to_txt_file(filename,var)
    
    
def run(mc,loc_data,conf_data,prior_boxes,prior_variances,gt_boxes_dense,gt_labels_dense,input_mask,all_match_overlaps):  
    with tf.Graph().as_default():
        loc_data_ = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_PRIORBOX, 4])
        conf_data_ = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_PRIORBOX, NUM_CLASSES])
        prior_boxes_ = tf.placeholder(tf.float32, [NUM_PRIORBOX, 4])
        prior_variances_ = tf.placeholder(tf.float32, [NUM_PRIORBOX, 4])
        
        gt_boxes_ = tf.placeholder(tf.float32, [BATCH_SIZE,NUM_PRIORBOX,4])
        gt_label_ = tf.placeholder(tf.float32, [BATCH_SIZE,NUM_PRIORBOX,mc.NUM_CLASSES])
        input_mask_ = tf.placeholder(tf.float32, [BATCH_SIZE,NUM_PRIORBOX])
        all_match_overlaps_ = tf.placeholder(tf.float32, [BATCH_SIZE,NUM_PRIORBOX])
        
        layer = MutiBoxLossLayer(mc)
        layer.process(loc_data_,conf_data_,prior_boxes_,prior_variances_,gt_boxes_,gt_label_,input_mask_,all_match_overlaps_)
        
        sess=tf.InteractiveSession()  
        sess.run(tf.global_variables_initializer())
    
        feed_dict = {
            loc_data_:loc_data,
            conf_data_:conf_data,
            prior_boxes_:prior_boxes,
            prior_variances_:prior_variances,
            gt_boxes_:gt_boxes_dense,
            gt_label_:gt_labels_dense,
            input_mask_:input_mask,
            all_match_overlaps_:all_match_overlaps
        }
        save_debug_var(sess,feed_dict,layer.debug_val,layer.debug_val_names)
        num_pos,num_neg,batch_neg_idx,gt_encode_boxes,loss = sess.run([layer.num_pos,layer.num_neg,
                                                                    layer.batch_neg_idx,
                                                                    layer.gt_encode_boxes,
                                                                    layer.loss,
                                                                    ],
                                                            feed_dict=feed_dict)
        print ('batch_neg_idx:',batch_neg_idx)
        print ('num_pos:',num_pos)
        #save_array_to_txt_file("gt_encode_boxes",gt_encode_boxes)
        print ('loss:',loss)
        
        
 
def parse_gt_data(mc,gt_data):
    gt_boxes = []
    gt_labels = []
    for i in range(0,BATCH_SIZE):
        gt_boxes.append([])
        gt_labels.append([])
        
    for i in range(0,len(gt_data)):
        batch_id = int(gt_data[i][0])
        assert batch_id < BATCH_SIZE
        gt_boxes[batch_id].append([gt_data[i][3],gt_data[i][4],gt_data[i][5],gt_data[i][6]])
        assert gt_data[i][1] < mc.NUM_CLASSES
        
        gt_labels[batch_id].append(gt_data[i][1])
        
    print ('[parse_gt_data]gt_boxes:',gt_boxes)
    print ('[parse_gt_data]gt_labels:',gt_labels)
    return gt_boxes,gt_labels
  
def ssd_iou(bbox1, bbox2):
    if bbox2[0] > bbox1[2] or bbox2[2] < bbox1[0] or bbox2[1] > bbox1[3] or bbox2[3] < bbox1[1]:
        return 0

    inter_xmin = max(bbox1[0], bbox2[0])
    inter_ymin = max(bbox1[1], bbox2[1])
    inter_xmax = min(bbox1[2], bbox2[2])
    inter_ymax = min(bbox1[3], bbox2[3])
    inter_width = inter_xmax - inter_xmin
    inter_height = inter_ymax - inter_ymin
    inter_size = inter_width * inter_height

    bbox1_size = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_size = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    return inter_size / (bbox1_size + bbox2_size - inter_size)

    
def MatchBBox(mc,prior_boxes,gt_boxes):
    all_match_indices = np.zeros((BATCH_SIZE,NUM_PRIORBOX)) - 1
    all_match_overlaps = np.zeros((BATCH_SIZE,NUM_PRIORBOX))
    
    for n in range(0,BATCH_SIZE):
    
        overlaps = np.zeros((NUM_PRIORBOX,len(gt_boxes[n])))
        for i in range(0,NUM_PRIORBOX):
            for j in range(0,len(gt_boxes[n])):
                overlap = ssd_iou(prior_boxes[i],gt_boxes[n][j])
                if overlap > 1e-6:
                    all_match_overlaps[n][i] = max(all_match_overlaps[n][i],overlap)
                    overlaps[i][j] = overlap
                     
        for i in range(0,len(gt_boxes[n])):
            max_anchor_idx = -1
            max_gt_idx = -1
            max_overlap = 0
            for j in range(0,NUM_PRIORBOX):
                if overlaps[j][i] > max_overlap:
                    max_anchor_idx = j
                    max_overlap = overlaps[j][i]
                    max_gt_idx = i
            all_match_overlaps[n][max_anchor_idx] =  max_overlap
            all_match_indices[n][max_anchor_idx] = max_gt_idx
            
        for i in range(0,NUM_PRIORBOX): 
            max_overlap = 0
            max_gt_idx = -1
            for j in range(0,len(gt_boxes[n])):
                if all_match_indices[n][i] != -1:
                    continue
                overlap = overlaps[i][j]
                if overlap > mc.overlap_threshold and overlap > max_overlap:
                    max_gt_idx = j
                    max_overlap = overlap
                    
            if  max_gt_idx != -1:    
                all_match_overlaps[n][i] = max_overlap
                all_match_indices[n][i] = max_gt_idx
                
    return all_match_indices,all_match_overlaps
   

def sparse_to_dense(mc,gt_boxes,gt_labels,all_match_indices):
    gt_boxes_dense = np.zeros([mc.BATCH_SIZE, mc.NUM_PRIORBOX,4])
    gt_labels_dense = np.zeros([mc.BATCH_SIZE, mc.NUM_PRIORBOX,mc.NUM_CLASSES])
    input_mask = np.zeros([mc.BATCH_SIZE, mc.NUM_PRIORBOX])

    for n in range(0,mc.BATCH_SIZE):
        for i in range(0,mc.NUM_PRIORBOX):
            one_label = np.zeros(NUM_CLASSES)
            one_box = np.zeros(4)
            if all_match_indices[n][i] == -1:
                one_label[mc.background_label_id] = 1 
            else:
                input_mask[n][i] = 1
                gt_idx = int(all_match_indices[n][i])
                label_idx = int(gt_labels[n][gt_idx])
                assert label_idx < mc.NUM_CLASSES
                one_label[label_idx] = 1
                one_box = gt_boxes[n][gt_idx]
            gt_labels_dense[n][i] = one_label
            gt_boxes_dense[n][i] = one_box
    
    return gt_boxes_dense,gt_labels_dense,input_mask
    
def decode_gt_box():
    pass
  
  
if __name__ == '__main__':
    mc = config()
   
    loc_data,conf_data,prior_data,gt_data = load_data(mc)
    save_array_to_txt_file("gt_data.txt",gt_data)
    
    gt_boxes,gt_labels = parse_gt_data(mc,gt_data)
    save_array_to_txt_file("prior_data.txt",prior_data)
    
    prior_boxes = np.reshape(prior_data[0:1,:,:],[NUM_PRIORBOX,4])
    save_array_to_txt_file("prior_boxes.txt",prior_boxes)
    
    prior_variances = np.reshape(prior_data[1:2,:,:],[NUM_PRIORBOX,4])
    save_array_to_txt_file("prior_variances.txt",prior_variances)
   
    all_match_indices,all_match_overlaps = MatchBBox(mc,prior_boxes,gt_boxes)
    gt_boxes_dense,gt_labels_dense,input_mask = sparse_to_dense(mc,gt_boxes,gt_labels,all_match_indices)
    save_array_to_txt_file("gt_labels_dense.txt",gt_labels_dense)
    save_array_to_txt_file("all_match_indices.txt",all_match_indices)
    save_array_to_txt_file("all_match_overlaps.txt",all_match_overlaps)
    save_array_to_txt_file("input_mask.txt",input_mask)
    run(mc,loc_data,conf_data,prior_boxes,prior_variances,gt_boxes_dense,gt_labels_dense,input_mask,all_match_overlaps)
    
    print ('==============run finsh!')
    
    
   