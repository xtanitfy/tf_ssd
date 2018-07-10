# Author:  08/25/2016

"""VGG16+ConvDet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton
import math
from mutibox_loss_layer import MutiBoxLossLayer

class SSDNet(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
    #with tf.device('/cpu:{}'.format(0)):
      print ('SSDNet __init__')
      ModelSkeleton.__init__(self, mc)
      self.mutibox_loss_layer = MutiBoxLossLayer(mc)
      self.add_forward_graph()
      self.add_interpretation_graph()
      self.add_loss_graph()
      self.add_train_graph()
    
  def add_forward_graph(self):

    print ('SSD _add_forward_graph')
    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)
    
      print ('load {} successful'.format(mc.PRETRAINED_MODEL_PATH))
      cw = self.caffemodel_weight

      scale = None
      for layername in cw:
        print (layername)

    self.paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]

    paddings = self.paddings 
    with tf.variable_scope('conv1') as scope:
      self._add_act(self.image_input,'data')  
      self.image = tf.pad(self.image_input,paddings,"CONSTANT")
      self.conv1_1 = self._conv_layer(
          'conv1_1', self.image, padding='VALID',filters=64, size=3, stride=1, freeze=True,xavier=True)
      print ('conv1_1:',self.conv1_1)
      
      self.conv1_1 = tf.pad(self.conv1_1,paddings,"CONSTANT")
      self.conv1_2 = self._conv_layer(
          'conv1_2', self.conv1_1, padding='VALID', filters=64, size=3, stride=1, freeze=True,xavier=True)
      
      self.pool1 = self._pooling_layer(
          'pool1', self.conv1_2, size=2, stride=2)
      
    with tf.variable_scope('conv2') as scope:
      self.pool1 = tf.pad(self.pool1,paddings,"CONSTANT")
      self.conv2_1 = self._conv_layer(
          'conv2_1', self.pool1, padding='VALID',filters=128, size=3, stride=1, freeze=True,xavier=True)

      self.conv2_1 = tf.pad(self.conv2_1,paddings,"CONSTANT")
      self.conv2_2 = self._conv_layer(
          'conv2_2', self.conv2_1,padding='VALID', filters=128, size=3, stride=1, freeze=True,xavier=True)
          
      self.pool2 = self._pooling_layer(
          'pool2', self.conv2_2, size=2, stride=2)
      
    with tf.variable_scope('conv3') as scope:
      self.pool2 = tf.pad(self.pool2,paddings,"CONSTANT")
      self.conv3_1 = self._conv_layer(
          'conv3_1', self.pool2,padding='VALID', filters=256, size=3, stride=1,xavier=True)
          
      self.conv3_1 = tf.pad(self.conv3_1,paddings,"CONSTANT")
      self.conv3_2 = self._conv_layer(
          'conv3_2', self.conv3_1,padding='VALID', filters=256, size=3, stride=1,xavier=True)
          
      self.conv3_2 = tf.pad(self.conv3_2,paddings,"CONSTANT")
      self.conv3_3 = self._conv_layer(
          'conv3_3', self.conv3_2,padding='VALID', filters=256, size=3, stride=1,xavier=True)
          
      self.pool3 = self._pooling_layer(
          'pool3', self.conv3_3, size=2, stride=2)
      
    with tf.variable_scope('conv4') as scope:
      self.pool3 = tf.pad(self.pool3,paddings,"CONSTANT")
      self.conv4_1 = self._conv_layer(
          'conv4_1', self.pool3,padding='VALID', filters=512, size=3, stride=1,xavier=True)

      self.conv4_1 = tf.pad(self.conv4_1,paddings,"CONSTANT")
      self.conv4_2 = self._conv_layer(
          'conv4_2', self.conv4_1,padding='VALID', filters=512, size=3, stride=1,xavier=True)

      self.conv4_2 = tf.pad(self.conv4_2,paddings,"CONSTANT")
      self.conv4_3 = self._conv_layer(
          'conv4_3', self.conv4_2,padding='VALID', filters=512, size=3, stride=1,xavier=True)
      self.pool4 = self._pooling_layer(
          'pool4', self.conv4_3, size=2, stride=2)
      
      #self.conv4_3_norm = self.l2_normalization("conv4_3_norm",self.conv4_3)
      #self._add_act(self.conv4_3_norm,'conv4_3_norm')
      
    with tf.variable_scope('conv5') as scope:
      self.pool4 = tf.pad(self.pool4,paddings,"CONSTANT")
      self.conv5_1 = self._conv_layer(
          'conv5_1', self.pool4,padding='VALID', filters=512, size=3, stride=1,xavier=True)

      self.conv5_1 = tf.pad(self.conv5_1,paddings,"CONSTANT")
      self.conv5_2 = self._conv_layer(
          'conv5_2', self.conv5_1,padding='VALID', filters=512, size=3, stride=1,xavier=True)

      self.conv5_2 = tf.pad(self.conv5_2,paddings,"CONSTANT")
      self.conv5_3 = self._conv_layer(
          'conv5_3', self.conv5_2,padding='VALID', filters=512, size=3, stride=1,xavier=True)

      self.conv5_3 = tf.pad(self.conv5_3,paddings,"CONSTANT")
      self.pool5 = self._pooling_layer(
          'pool5', self.conv5_3, size=3, stride=1,padding='VALID')     

    self.pool5 = tf.pad(self.pool5,[[0, 0], [6, 6], [6, 6], [0, 0]],"CONSTANT")
    self.fc6 = self._conv_layer(
          'fc6', self.pool5,padding='VALID', dilation=6,filters=1024, size=3, stride=1,xavier=True)   
    
    self.fc7 = self._conv_layer(
    'fc7', self.fc6,padding='VALID', filters=1024, size=1, stride=1,xavier=True)

    with tf.variable_scope('conv6') as scope:
      self.conv6_1 = self._conv_layer(
          'conv6_1', self.fc7,padding='VALID', filters=256, size=1, stride=1,xavier=True)
      
      self.conv6_1 = tf.pad(self.conv6_1,paddings,"CONSTANT")
      self.conv6_2 = self._conv_layer(
          'conv6_2', self.conv6_1,padding='VALID', filters=512, size=3, stride=2,xavier=True)
      
    with tf.variable_scope('conv7') as scope:
      self.conv7_1 = self._conv_layer(
          'conv7_1', self.conv6_2,padding='VALID', filters=128, size=1, stride=1,xavier=True)
      self.conv7_1 = tf.pad(self.conv7_1,paddings,"CONSTANT")
      self.conv7_2 = self._conv_layer(
          'conv7_2', self.conv7_1,padding='VALID', filters=256, size=3, stride=2,xavier=True)

    with tf.variable_scope('conv8') as scope:
      self.conv8_1 = self._conv_layer(
          'conv8_1', self.conv7_2,padding='VALID', filters=128, size=1, stride=1,xavier=True)
      self.conv8_2 = self._conv_layer(
          'conv8_2', self.conv8_1,padding='VALID', filters=256, size=3, stride=1,xavier=True)     
    
    with tf.variable_scope('conv9') as scope:
      self.conv9_1 = self._conv_layer(
        'conv9_1', self.conv8_2,padding='VALID', filters=128, size=1, stride=1,xavier=True)
      self.conv9_2 = self._conv_layer(
        'conv9_2', self.conv9_1,padding='VALID', filters=256, size=3, stride=1,xavier=True)
      
    mc.mbox_source_layers = [self.conv4_3,self.fc7,self.conv6_2,
                                    self.conv7_2,self.conv8_2,self.conv9_2]
    self.mbox_confs = []
    self.mbox_locs = []
    self.mbox_priorboxs = []
    for i, name in enumerate(mc.mbox_source_layers): 
      self.multibox_layer(mc.mbox_source_layers[i],i)
    
    self.mbox_conf = tf.concat(self.mbox_confs, 1, name='mbox_conf')
    self._add_act(self.mbox_conf,'mbox_conf')
    
    self.mbox_loc = tf.concat(self.mbox_locs, 1, name='mbox_loc')
    self._add_act(self.mbox_loc,'mbox_loc')
    
    print ('SSDNet _add_forward_graph finish!') 


  def multibox_layer(self,base_layer,src_idx):
    mc = self.mc

    src_layername = mc.mbox_source_layers_name[src_idx]
    if mc.normalizations[src_idx] != -1:
      init_scale = mc.normalizations[src_idx]
      output_layername = src_layername + '_norm'
      self.base = self.l2_normalization(output_layername,base_layer,init_var=init_scale)
    else:
      output_layername = src_layername
      self.base = base_layer
    self._add_act(self.base,output_layername)
    
    min_size = mc.min_sizes[src_idx]
    max_size = mc.max_sizes[src_idx]
    aspect_ratio = mc.aspect_ratios[src_idx]
    if max_size:
      num_priors_per_location = 2 + len(aspect_ratio)
    else:
      #only min_size
      num_priors_per_location = 1 + len(aspect_ratio)
      
    #aspect_ratio:[1,1]
    num_priors_per_location += len(aspect_ratio)

    paddings = self.paddings
    
    with tf.variable_scope(output_layername+'mbox_loc') as scope: 
      name = "{}_mbox_loc".format(output_layername)
      num_loc_output = num_priors_per_location * 4;
      self.base_tmp = tf.pad(self.base,paddings,"CONSTANT")
      mbox_loc = self._conv_layer(
          name, self.base_tmp,padding='VALID', filters=num_loc_output, size=3, stride=1,relu=False)
      self._add_act(mbox_loc,name)
      
      flat_name = "{}_flat".format(name)
      mbox_loc_flat = tf.reshape(mbox_loc,[mc.BATCH_SIZE,-1])
      self._add_act(mbox_loc_flat,flat_name)

      self.mbox_locs.append(mbox_loc_flat)
      
    with tf.variable_scope(output_layername+'_mbox_conf') as scope:
      name = "{}_mbox_conf".format(output_layername)
      num_conf_output = num_priors_per_location * (len(mc.CLASS_NAMES));
      self.base_tmp = tf.pad(self.base,paddings,"CONSTANT")
      mbox_conf = self._conv_layer(
          name, self.base_tmp,padding='VALID', filters=num_conf_output, size=3, stride=1,relu=False)
      self._add_act(mbox_conf,name)

      flat_name = "{}_flat".format(name)
      mbox_conf_flat = tf.reshape(mbox_conf,[mc.BATCH_SIZE,-1])
      self._add_act(mbox_conf_flat,flat_name)

      self.mbox_confs.append(mbox_conf_flat)
    

  def decode_box(self,prior_bboxes,prior_variances):
    mc = self.mc
    print ('self.mbox_loc:',self.mbox_loc)
    mbox_loc_reshape = tf.reshape(self.mbox_loc,[mc.BATCH_SIZE,-1,4])
    delta_xmin, delta_ymin, delta_xmax, delta_ymax = tf.unstack(
            mbox_loc_reshape, axis=2)
    prior_bboxes_reshape = tf.reshape(prior_bboxes,[-1,4])
    prior_variances_reshape = tf.reshape(prior_variances,[-1,4])
    
    prior_width = prior_bboxes_reshape[:,2] - prior_bboxes_reshape[:,0]
    prior_height = prior_bboxes_reshape[:,3] - prior_bboxes_reshape[:,1]
    prior_center_x = (prior_bboxes_reshape[:,0] + prior_bboxes_reshape[:,2])/2.
    prior_center_y = (prior_bboxes_reshape[:,1] + prior_bboxes_reshape[:,3])/2.
    
    bbox_center_x = tf.identity(prior_variances_reshape[:,0] * delta_xmin * prior_width + prior_center_x)
    bbox_center_y = tf.identity(prior_variances_reshape[:,1] * delta_ymin * prior_height + prior_center_y) 
    bbox_width = tf.identity(util.safe_exp(prior_variances_reshape[:,2] *delta_xmax, mc.EXP_THRESH) * prior_width)
    bbox_height = tf.identity(util.safe_exp(prior_variances_reshape[:,3] *delta_ymax, mc.EXP_THRESH) * prior_height)
    xmins, ymins, xmaxs, ymaxs = util.bbox_transform(
        [bbox_center_x, bbox_center_y, bbox_width, bbox_height])
    '''
    xmins = tf.minimum(
        tf.maximum(0.0, xmins), 1., name='bbox_xmin')
    ymins = tf.minimum(
        tf.maximum(0.0, ymins), 1., name='bbox_ymin')
    xmaxs = tf.maximum(
        tf.minimum(1., xmaxs), 0.0, name='bbox_xmax')
    ymaxs = tf.maximum(
        tf.minimum(1., ymaxs), 0.0, name='bbox_ymax')
    '''
    xmins *= mc.IMAGE_WIDTH
    xmaxs *= mc.IMAGE_WIDTH
    ymins *= mc.IMAGE_HEIGHT
    ymaxs *= mc.IMAGE_HEIGHT
    
    self.decode_boxes = tf.stack([xmins, ymins, xmaxs, ymaxs],axis=2)
    self._add_act(self.decode_boxes,'decode_boxes')
    
        
  def add_interpretation_graph(self):
    mc = self.mc
    self.mbox_pred = self.mbox_conf
    self.mbox_conf = tf.reshape(
                        tf.nn.softmax(
                            tf.reshape(self.mbox_conf,
                            [-1,len(mc.CLASS_NAMES)]),
                            ),
                            [mc.BATCH_SIZE,-1,len(mc.CLASS_NAMES)],
                            name = 'mbox_conf_flatten'
                        )
    self._add_act(self.mbox_conf,'mbox_conf')

    print ('self.mbox_conf:',self.mbox_conf)
    
    #prior_bboxes = self.mbox_priorbox[0:1,:]
    #prior_variances = self.mbox_priorbox[1:,:]
    prior_bboxes = tf.convert_to_tensor(mc.ANCHOR_BOX,dtype=tf.float32)
    prior_variances = tf.convert_to_tensor(mc.PRIORBOX_VARIANCES,dtype=tf.float32)
    
    self.decode_box(prior_bboxes,prior_variances)
    
    print ('ssd _add_interpretation_graph')
    class_num = len(mc.CLASS_NAMES)
    
    self.mbox_conf = tf.reshape(self.mbox_conf,[mc.BATCH_SIZE,-1,class_num])
    self.finnal_mbox_conf = tf.reduce_max(self.mbox_conf, 2, name='score')
    self.det_class_idx = tf.argmax(self.mbox_conf, 2, name='class_idx')
    
    print ('self.decode_boxes:',self.decode_boxes)
    print ('self.mbox_conf:',self.mbox_conf)
    #print ('self.det_class_idx:',self.det_class_idx)
    #raw_input('Enter and continue')

    #self.loss = tf.get_collection('losses')

    
  def add_loss_graph(self):
    print ('self.mbox_conf')
    mc = self.mc
    self.mbox_loc_reshape = tf.reshape(self.mbox_loc,[mc.BATCH_SIZE,-1,4])
    self.mutibox_loss_layer.process(self.mbox_loc_reshape,self.mbox_conf,self.mbox_pred,
                                        self.gt_boxes_,self.gt_label_,
                                          self.input_mask_,self.all_match_overlaps_)
    tf.add_to_collection('losses', self.mutibox_loss_layer.loss)
    self.losses = tf.add_n(tf.get_collection('losses'), name='total_loss')
    
  def add_train_graph(self):
    
    mc = self.mc
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    self.lr = self._configure_learning_rate(mc,self.global_step)
    '''
    self.lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                      self.global_step,
                      mc.DECAY_STEPS,
                      mc.LR_DECAY_FACTOR,
                      staircase=True)
    '''
    
    optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=mc.MOMENTUM)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
      grads_vars = optimizer.compute_gradients(self.losses,tf.trainable_variables())
      with tf.variable_scope('clip_gradient') as scope:
        for i, (grad, var) in enumerate(grads_vars):
          #print ("2--var.name:",var.op.name)
          if mc.CLIP_GRAD == True:
            grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)
          else:
            pass
      
      apply_gradient_op = optimizer.apply_gradients(grads_vars,global_step=self.global_step)
      with tf.control_dependencies([apply_gradient_op]):
        self.train_op = tf.no_op(name='train')
   
