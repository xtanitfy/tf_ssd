# -*- coding: utf-8 -*-
# Author:  08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf
from struct import *
from config import *
from train import _draw_box
from nets import *
from data_layer import Data_layer


FLAGS = tf.app.flags.FLAGS

debug_filename = 'debug_txt'

#tf.app.flags.DEFINE_string('net', 'SSD',
#                           """Neural net architecture.""")
                           
tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")

                           
#./data/model_checkpoints/squeezeDet/model.ckpt-87000
tf.app.flags.DEFINE_string(
    'checkpoint', './SqueezeDet/model_save/model.ckpt-99000',
    """Path to the model parameter file.""")
    
tf.app.flags.DEFINE_string(
    'input_path', 'data/s48.jpg',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")

tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")

def preprocess_frame(mc,frame):
    rows,cols,channels = frame.shape
    if cols < mc.IMAGE_WIDTH or rows < mc.IMAGE_HEIGHT:
        print ("cols < mc.IMAGE_WIDTH or rows < mc.IMAGE_HEIGHT")
        return frame,-1
    else:
        ystart = (rows - mc.IMAGE_HEIGHT)/2
        xstart = (cols - mc.IMAGE_WIDTH)/2
        frame = frame[ystart:ystart+mc.IMAGE_HEIGHT,xstart:xstart+mc.IMAGE_WIDTH, :]
        return frame,0
        
    
    
def video_demo():
  """Detect videos."""

  cap = cv2.VideoCapture(FLAGS.input_path)
 # cap = cv2.VideoCapture(0)
  print ("FLAGS.input_path:",FLAGS.input_path)
  if False == cap.isOpened():  
    print ('open video failed')
    return 
  else:  
    print ('open video succeeded')
        
  # Define the codec and create VideoWriter object
  # fourcc = cv2.cv.CV_FOURCC(*'XVID')
  fourcc = cv2.cv.CV_FOURCC('X','V','I','D')
  # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
  # in_file_name = os.path.split(FLAGS.input_path)[1]
  # out_file_name = os.path.join(FLAGS.out_dir, 'out_'+in_file_name)
  out = cv2.VideoWriter("res.avi", fourcc, 30.0, (1242,375), True)
 #  out = VideoWriter(out_file_name, frameSize=(1242, 375))
  #out.open()
  numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
  
  with tf.Graph().as_default():
    # Load model
    mc = kitti_squeezeDet_config()
    mc.BATCH_SIZE = 1
    # model parameters will be restored from checkpoint
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDet(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)
    cnt = 0
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      times = {}
      count = 0
      while cap.isOpened():
        #print ("open success!")
        
        t_start = time.time()
        count += 1
        out_im_name = os.path.join(FLAGS.out_dir, str(count).zfill(6)+'.jpg')
        
        # Load images from video and crop
        ret, frame = cap.read()
      #  print ("np.shape(frame):",np.shape(frame))
        if ret==True:
          # crop frames
          #frame = frame[500:-205, 239:-439, :]
          frame,ret= preprocess_frame(mc,frame)
          if ret < 0:
            print ("preprocess_frame error")
            return 
            
          im_input = frame.astype(np.float32) - mc.BGR_MEANS
        else:
          break
        
        t_reshape = time.time()
        times['reshape']= t_reshape - t_start

        # Detect
        det_boxes, det_probs, det_class = sess.run(
            [model.det_boxes, model.det_probs, model.det_class],
            feed_dict={model.image_input:[im_input], model.keep_prob: 1.0})

        t_detect = time.time()
        times['detect']= t_detect - t_reshape
        
        # Filter
        final_boxes, final_probs, final_class = model.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])

        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        t_filter = time.time()
        times['filter']= t_filter - t_detect

        # Draw boxes

        # TODO(bichen): move this color dict to configuration file
        cls2clr = {
            'car': (255, 191, 0),
            'cyclist': (0, 191, 255),
            'pedestrian':(255, 0, 191)
        }
        _draw_box(
            frame, final_boxes,
            [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                for idx, prob in zip(final_class, final_probs)],
            cdict=cls2clr
        )

        t_draw = time.time()
        times['draw']= t_draw - t_filter

        cv2.imwrite(out_im_name, frame)
        out.write(frame)
        cv2.imshow("show",frame)
        
        times['total']= time.time() - t_start

        # time_str = ''
        # for t in times:
        #   time_str += '{} time: {:.4f} '.format(t[0], t[1])
        # time_str += '\n'
        time_str = 'Total time: {:.4f}, detection time: {:.4f}, filter time: '\
                   '{:.4f}'. \
            format(times['total'], times['detect'], times['filter'])

        #print (time_str)

        if (cnt % 100 == 0):
            print ("cnt:",cnt," allframes:",numFrames)
            
        cnt = cnt + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  # Release everything if job is finished
  cap.release()
  out.release()
  cv2.destroyAllWindows()

def save_array_to_txt_file(filename,arr):
    all_size = 1
    for i in range(0,len(arr.shape)):
        all_size *=  arr.shape[i]
    print ("all_size:",all_size)   
    
    act_flat = arr.reshape(all_size)
    
    print ('save ',filename)
    filename = filename.split('.')[-2] + '.txt'
    file = open(filename, "w")
    cnt = 0
    for i in range(0,all_size):
        str = '%-12f' % act_flat[i]
        file.write(str)
        if (cnt % 64 == 0 and cnt != 0):
            file.write('\n')
        cnt = cnt + 1
    file.close()
    
def save_array_to_binfile(filename,data):
    
    filename = filename.split('.')[-2] + '.bin'
    file = open(filename, "wb")
    
    all_size = 1
    for i in range(0,len(data.shape)):
        all_size *= data.shape[i]
        
    act_flat = data.reshape(all_size)
    
    for i in range(0,act_flat.shape[0]):
        file.write(pack("f",float(act_flat[i])))
    file.close()

def save_debug_val():
    debugs_values = get_debug_val();
    debugs_values_name = get_debug_val_name()
    global debug_filename
    for i in range(0,len(debugs_values)):
        filename = '{}/{}.txt'.format(debug_filename,debugs_values_name[i])
        save_array_to_txt_file(filename,debugs_values[i])
        

def save_input_image_data(input_image):
    filename = 'debug_txt/data_0.txt'
    file = open(filename, "w")

    all_size = 1
    for i in range(0,len(input_image.shape)):
        all_size *=  input_image.shape[i]
    print ("all_size:",all_size)   

    input_image_new = np.transpose(np.array(input_image),(2,0,1))
    data_reshape = input_image_new.reshape(all_size)
    
    cnt = 0
    for i in range(0,all_size):
        str = '%.8f ' % data_reshape[i]
        file.write(str)
        if (cnt % 64 == 0 and cnt != 0):
            file.write(' \n')
        cnt = cnt + 1
    file.close()

def save_tf_conv12(sess,model,input_image,activation,name):
    act = sess.run( activation,
            feed_dict={model.image_input:[input_image]})

    print (name + ' shape:' + str(act.shape))
    global debug_filename
    
    filename = '{}/tf_{}.txt'.format(debug_filename,name)  
    
    save_array_to_txt_file(filename,act)
    save_array_to_binfile(filename,act)

def save_activations(sess,model,input_image,activation,name):
    act = sess.run( activation,
            feed_dict={model.image_input:[input_image]})

    #act_new = []
    if (len(act.shape) == 4):
        act_new = np.transpose(act,(0,3,1,2))    
    else:
        print ('name shape is not 4')
        act_new = act

    print (name + ' shape:' + str(act_new.shape))
    global debug_filename
    
    filename = '{}/{}'.format(debug_filename,name)
    '''
    for i in range(0,len(act_new.shape)):
        if act_new.shape[i] != 1:
            filename += '_' + str(act_new.shape[i])
    '''
    filename += '.txt'    
    
    save_array_to_txt_file(filename,act_new)
    #save_array_to_binfile(filename,act_new)

def save_mobiledet_params(sess,kernel_name='weights',bias_name='biases'):
  all_vars = tf.all_variables()
  type = ''
  for v in all_vars:
    if kernel_name in v.name or bias_name in v.name or 'BatchNorm' in v.name:
      if 'Momentum' in v.name:
        continue
      val = v.eval(sess)

      if kernel_name in v.name:
        type = 'kernels'
        
      elif bias_name in v.name:
        type = 'bias'

      elif 'BatchNorm' in v.name:
        type = 'batchnorm-' + v.name[0:len(v.name)-2].split('/')[3]

      if 'conv12' in v.name:
        layername = 'conv12'
      else:
        layername = v.name[0:len(v.name)-2].split('/')[1]
        
      newname = layername + '-' + type
      #print (v.name,val.shape)
      
      #newname = v.name[0:len(v.name)-2].replace('/','-')

      print ('=========>==',newname)
      filename = ""
      
      if 'kernels' in v.name:
        val_new = np.transpose(val,(3,2,0,1))
        filename = 'weights_bin/{}.bin'.format(newname)
        val_tmp = val_new
      else:
        filename = 'weights_bin/{}.bin'.format(newname)
        val_tmp = val
      
      save_array_to_binfile(filename,val_tmp)
      save_array_to_txt_file(filename,val_tmp)    

def save_weights(sess,kernel_name='kernels',bias_name='biases'):
  all_vars = tf.all_variables()
  for v in all_vars:
    if kernel_name in v.name or bias_name in v.name or 'BatchNorm' in v.name:
      if 'Momentum' in v.name:
        continue
      val = v.eval(sess)
      #print (v.name,val)
      print (v.name,val.shape)
      
      newname = v.name[0:len(v.name)-2].replace('/','-')

      #print (newname)
      filename = ""
      '''
      if 'kernels' in v.name:
        val_new = np.transpose(val,(3,2,0,1))
        filename = 'weights_bin/w_{}_{}_{}_{}_{}.bin'.format(newname,
            val_new.shape[0],val_new.shape[1],val_new.shape[2],val_new.shape[3])
        val_tmp = val_new 
  
      else:
        filename = 'weights_bin/b_{}_{}.bin'.format(newname,val.shape[0])
        val_tmp = val
      '''
      
      if 'kernels' in v.name:
        val_new = np.transpose(val,(3,2,0,1))
        filename = 'weights_bin/{}.bin'.format(newname)
        val_tmp = val_new
      else:
        filename = 'weights_bin/{}.bin'.format(newname)
        val_tmp = val
      
      save_array_to_binfile(filename,val_tmp)
      save_array_to_txt_file(filename,val_tmp)

def filter_variables(all_variables):
    left_variables = []

    print ('filter_variables')
    filter_key = ['Momentum','iou','global_step']
    for v in all_variables:
      
      print ("0----name:",v.op.name)

      isexist = False
      for i in range(0,len(filter_key)):
        if filter_key[i] in v.name:
            isexist = True
            break
      if isexist == True:
        continue
        
      left_variables += [v]

    return left_variables


def draw_image_annno(image,gt_boxes,filename):
    height, width, _ = [int(v) for v in image.shape]
    #print (filename,' height:',height,' width:',width)
    
    for box in gt_boxes:
        box_1 = [int(box[i]) for i in range(0,len(box))]
        cv2.rectangle(image, (box_1[0], box_1[1]), (box_1[2], box_1[3]), (255,0,0), 2)
   
    cv2.imwrite(filename,image)
        
def image_demo():
  """Detect image."""

  with tf.Graph().as_default():
    # Load model
    
    #mc.is_training = False
    if FLAGS.net == 'squeezeDet':
      print ('============squeezeDet')
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      mc.is_training = False
      model = SqueezeDet(mc, FLAGS.gpu)
      saver = tf.train.Saver(model.model_params)
      
    elif FLAGS.net == 'mobileDet':
      print ('============mobileDet')
      mc = kitti_mobileDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      mc.is_training = False
      model = MobileDet(mc, FLAGS.gpu)
      gblobal_variables = filter_variables(tf.global_variables())
      saver = tf.train.Saver(gblobal_variables)
      
    elif FLAGS.net == 'mobileDet_V1_025':
      print ('============mobileDet_V1_025')
      mc = kitti_mobileDet_V1_025_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      mc.is_training = False
      model = MobileDet(mc, FLAGS.gpu)
      gblobal_variables = filter_variables(tf.global_variables())
      saver = tf.train.Saver(gblobal_variables)  

    elif FLAGS.net == 'SSD':
      mc = vkitti_SSD_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = True
      #print ('FLAGS.pretrained_model_path:',FLAGS.pretrained_model_path)
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      mc.is_training = False
      model = SSDNet(mc,FLAGS.gpu)
      #saver = tf.train.Saver(tf.global_variables())

    else:
      print ('No this net type!')
      return
 
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      init = tf.global_variables_initializer()
      sess.run(init)
    
      #print ('restore:',FLAGS.checkpoint)
      #saver.restore(sess, FLAGS.checkpoint)

      for f in glob.iglob(FLAGS.input_path):
        im = cv2.imread(f)
        im_bak = im
        #im = im.astype(np.float32, copy=False) 
        src_h, src_w, _ = [float(v) for v in im.shape]
        
        print ('mc.BGR_MEANS:',mc.BGR_MEANS)
        im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
        
        im = im - mc.BGR_MEANS
        input_image = im

        #save_input_image_data(im)
        # Detect
        orig_h, orig_w, _ = [float(v) for v in im.shape]
        print ('image rows:{} image cols:{}'.format(orig_h,orig_w))
        #raw_input('Enter and continue!')

        
        #conv1_1 = sess.run([model.conv1_1],feed_dict={model.image_input:[input_image]})
        det_boxes, det_probs = sess.run(
            [model.decode_boxes, model.mbox_conf],
            feed_dict={model.image_input:[input_image]})

        det_probs[:,:,0] = 0
        probs = det_probs
        det_probs = np.max(probs,2)
        det_class = np.argmax(probs,2)
        
        save_array_to_txt_file('_det_probs.txt',det_probs)
        save_array_to_txt_file('_det_class.txt',det_class)
        
        #raw_input('pasue')
        # Filter

        #这里因为预测的是(xmin,ymin,xmax,ymax) 而iou函数计算的是(cx,cy,h,w)

        final_boxes, final_probs, final_class = model.ssd_filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])

        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        
        final_boxes_arr = np.array(final_boxes)
        scale_width = float(src_w) / float(orig_w)
        scale_height = float(src_h) / float(orig_h)
        final_boxes_arr[:,0::2] *=  scale_width
        final_boxes_arr[:,1::2] *=  scale_height
        print ("final_boxes:",final_boxes_arr)
        print ("final_probs:",final_probs)
        print ("final_class:",final_class)
        
        draw_image_annno(im_bak,final_boxes_arr,"draw.jpg")
        
        
        '''
        # TODO(bichen): move this color dict to configuration file
        cls2clr = {
            'car': (255, 191, 0),
            'cyclist': (0, 191, 255),
            'pedestrian':(255, 0, 191)
        }

        # Draw boxes
        _draw_box(
            im, final_boxes,
            [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                for idx, prob in zip(final_class, final_probs)],
            cdict=cls2clr,
        )

        file_name = os.path.split(f)[1]
        out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)
        cv2.imwrite(out_file_name, im)
        print ('Image detection output saved to {}'.format(out_file_name))
        '''

        
        for i in range(0,len(model.act)):
          print ('save -- model.act_names:',model.act_names[i]) 
          #if 'decode_boxes' == model.act_names[i]:
          save_activations(sess,model,input_image,model.act[i],model.act_names[i])
        
        
        '''
        if FLAGS.net == 'mobileDet':
          save_mobiledet_params(sess,kernel_name='weights')
          
        elif FLAGS.net == 'squeezeDet':
          save_weights(sess)
        
        save_debug_val()
        '''
        
def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
    
  if FLAGS.mode == 'image':
    start = time.clock()
    image_demo()
    end = time.clock()
    print ("comsume time: %f s" % (end - start))
    
  else:
    video_demo()

if __name__ == '__main__':
    tf.app.run()
