# -*- coding: utf-8 -*-
# Author:shawn  11/25/2017

"""Train"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf
import threading
import shutil 
from config import *
from dataset import pascal_voc, kitti
from utils.util import  bgr_to_rgb, bbox_transform
from nets import *
from data_layer import Data_layer
import signal

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently only support KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('year', '2007',
                            """VOC challenge year. 2007 or 2012"""
                            """Only used for Pascal VOC dataset""")

tf.app.flags.DEFINE_string('train_dir', '/disk3/tf_workspace/squeezeDet-master/logs/squeezedet/train',
                            """Directory where to write event logs """
                            """and checkpoint.""")

tf.app.flags.DEFINE_string('restore_dir', '/disk3/tf_workspace/squeezeDet-master/logs/squeezedet/train',
                            """restore_dir"""
                            """and checkpoint.""")                         
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Maximum number of batches to run.""")
tf.app.flags.DEFINE_string('net', 'squeezeDet',
                           """Neural net architecture. """)
tf.app.flags.DEFINE_string('pretrained_model_path', '',
                           """Path to the pretrained model.""")
tf.app.flags.DEFINE_integer('summary_step', 10,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 100,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

tf.app.flags.DEFINE_integer('resume', '1', """resume""")

bstop = False

def filter_variables(all_variables):
    left_variables = []

    filter_key = ['Momentum','iou','global_step']
    #'Conv2d_12_depthwise',
    #'ExponentialMovingAverage','pointwise/biases','depthwise/biases','Conv2d_0/biases'
    for v in all_variables:
      print ("name:",v.name)

      isexist = False
      for i in range(0,len(filter_key)):
        if filter_key[i] in v.name:
            isexist = True
            break
      if isexist == True:
        continue
        
      left_variables += [v]

    return left_variables

def myHandler(signum, frame):
    global bstop
    print('I received: ', signum)
    bstop = True


def save_model(sess,saver,step):
    global bstop
    checkpoint_path = os.path.join(FLAGS.restore_dir, 'model.ckpt')
    if bstop == True:
      print ('Stop and save ',checkpoint_path)
    
    saver.save(sess, checkpoint_path, global_step=step)
    
    model_file = FLAGS.restore_dir + "/model.ckpt-" + str(step) + ".data-00000-of-00001"
    shutil.copy(model_file,FLAGS.train_dir+"/../backup")
    
    model_file = FLAGS.restore_dir + "/model.ckpt-" + str(step) + ".index"
    shutil.copy(model_file,FLAGS.train_dir+"/../backup")
    
    model_file = FLAGS.restore_dir + "/model.ckpt-" + str(step) + ".meta"
    shutil.copy(model_file,FLAGS.train_dir+"/../backup")

def ssd_train():
    global bstop
    assert FLAGS.dataset == 'KITTI' or FLAGS.dataset == 'VKITTI', \
      'Currently only support KITTI dataset'

    signal.signal(signal.SIGINT, myHandler)
    
    print ('FLAGS.PRETRAINED_MODEL_PATH:',FLAGS.pretrained_model_path)
    
    with tf.Graph().as_default():
      assert  FLAGS.net == 'SSD'
      
      if FLAGS.net == 'SSD':
          mc = vkitti_SSD_config()
          mc.LOAD_PRETRAINED_MODEL = True
          mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
          mc.is_training = True
          model = SSDNet(mc,FLAGS.gpu)
          print ('SSD net')

      data_layer = Data_layer(FLAGS.image_set, FLAGS.data_path, mc)
      #if bresume == False:
      #    gblobal_variables = filter_variables(tf.global_variables())
      #    saver1 = tf.train.Saver(gblobal_variables)
      saver = tf.train.Saver(tf.global_variables())
      
      init = tf.global_variables_initializer()
      sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
      sess.run(init)
      tf.train.start_queue_runners(sess=sess)

      
      history_step = 0
      ckpt = tf.train.get_checkpoint_state(FLAGS.restore_dir)
      if ckpt and ckpt.model_checkpoint_path:
          print ("===>FLAGS.restore:",ckpt.model_checkpoint_path) 
          ret = saver.restore(sess, ckpt.model_checkpoint_path)
          history_step = int(ckpt.model_checkpoint_path.split('-')[-1])

      def load_data():
        batch_image,gt_boxes_dense,gt_labels_dense,input_mask,all_match_overlaps = data_layer.Get_feed_data()
        #add queue
        feed_dict = { model.ph_image_input        :batch_image,
                      model.ph_gt_boxes_          :gt_boxes_dense,
                      model.ph_gt_label_          :gt_labels_dense,
                      model.ph_input_mask_        :input_mask,
                      model.ph_all_match_overlaps_:all_match_overlaps}  
        return feed_dict
        
      def _enqueue(sess, coord):
        try:
          while not coord.should_stop(): 
            if bstop == True:
              break
              
            feed_dict = load_data()
            sess.run(model.enqueue_op, feed_dict=feed_dict)
          print ('Enqueue finish!')
          
        except Exception, e:
          coord.request_stop(e)

      coord = tf.train.Coordinator()
      if mc.NUM_THREAD > 0:
        enq_threads = []
        for _ in range(mc.NUM_THREAD):
          enq_thread = threading.Thread(target=_enqueue, args=[sess, coord])
          # enq_thread.isDaemon()
          enq_thread.start()
          enq_threads.append(enq_thread)
        
      threads = tf.train.start_queue_runners(coord=coord, sess=sess)
      run_options = tf.RunOptions(timeout_in_ms=60000)
      
      for step in xrange(history_step,FLAGS.max_steps):
          if bstop == True:
              #sess.run(model.FIFOQueue.close(cancel_pending_enqueues=True))
              #coord.request_stop()
              coord.join(threads)
              save_model(sess,saver,step)
              break
              
          start_time = time.time() 
         
          #feed_dict = load_data()             
          op_list = [model.decode_boxes, 
                     model.mbox_conf, 
                     model.det_class_idx,
                     model.lr,
                     model.mutibox_loss_layer.loc_loss,
                     model.mutibox_loss_layer.conf_loss,
                     model.losses,
                     model.train_op]

          losses = 0
          det_boxes, det_probs, det_class,lr,loc_loss,conf_loss,losses,_ = sess.run(op_list,options=run_options)
          duration = time.time() - start_time
          
          if step % 10 == 0:
              num_images_per_step = mc.BATCH_SIZE
              images_per_sec = num_images_per_step / duration
              sec_per_batch = float(duration)
              print ('%s: step %d, loss=%.6f loc_loss=%.6f conf_loss=%.6f lr=%.6f %.3f sec/batch)' % (
                                      datetime.now(),
                                      step,
                                      losses,
                                      loc_loss,
                                      conf_loss,
                                      lr,
                                      sec_per_batch))

          if step == 0:
              continue
              
          if step % FLAGS.checkpoint_step == 0 or step == FLAGS.max_steps:
              save_model(sess,saver,step)

            
def main(argv=None): 
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  ssd_train()

if __name__ == '__main__':
  tf.app.run()


