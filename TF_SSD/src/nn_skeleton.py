# Author:  08/25/2016
# -*- coding: utf-8 -*- 
"""Neural network model base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
#from binary_ops import * 
from dorefa import get_dorefa

import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope

def _add_loss_summaries(total_loss):
  """Add summaries for losses
  Generates loss summaries for visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  """
  losses = tf.get_collection('losses')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    tf.summary.scalar(l.op.name, l)

def _variable_on_device(name, shape, initializer, trainable=True):
  """Helper to create a Variable.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  # TODO(bichen): fix the hard-coded data type below
  dtype = tf.float32
  if not callable(initializer):
    var = tf.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_device(name, shape, initializer, trainable)
  if wd is not None and trainable:
    #print (name + ' add weight_decay loss')
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

class ModelSkeleton:
  """Base class of NN detection models."""
  act = []
  act_names = []
  debug_val = []
  debug_val_names = []
  
  def __init__(self, mc):
    self.mc = mc
    
    
    # image batch input
    self.ph_image_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],
        name='image_input'
    )
    # a scalar tensor in range (0, 1]. Usually set to 0.5 in training phase and
    # 1.0 in evaluation phase
    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # A tensor where an element is 1 if the corresponding box is "responsible"
    # for detection an object and 0 otherwise.
    self.ph_input_mask = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, 1], name='box_mask')
    # Tensor used to represent bounding box deltas.
    self.ph_box_delta_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, 4], name='box_delta_input')
    # Tensor used to represent bounding box coordinates.
    self.ph_box_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, 4], name='box_input')
    # Tensor used to represent labels
    self.ph_labels = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES], name='labels')

    # Tensor representing the IOU between predicted bbox and gt bbox
    self.ious = tf.Variable(
        initial_value=np.zeros((mc.BATCH_SIZE, mc.ANCHORS)), trainable=False,
        name='iou', dtype=tf.float32
    )

    self.FIFOQueue = tf.FIFOQueue(
        capacity=mc.QUEUE_CAPACITY,
        dtypes=[tf.float32, tf.float32, tf.float32, 
                tf.float32, tf.float32],
        shapes=[[mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],
                [mc.ANCHORS, 1],
                [mc.ANCHORS, 4],
                [mc.ANCHORS, 4],
                [mc.ANCHORS, mc.CLASSES]],
    )

    self.enqueue_op = self.FIFOQueue.enqueue_many(
        [self.ph_image_input, self.ph_input_mask,
         self.ph_box_delta_input, self.ph_box_input, self.ph_labels]
    )

    self.image_input, self.input_mask, self.box_delta_input, \
        self.box_input, self.labels = tf.train.batch(
            self.FIFOQueue.dequeue(), batch_size=mc.BATCH_SIZE,
            capacity=mc.QUEUE_CAPACITY) 
    
    # model parameters
    self.model_params = []

    # model size counter
    self.model_size_counter = [] # array of tuple of layer name, parameter size
    # flop counter
    self.flop_counter = [] # array of tuple of layer name, flop number
    # activation counter
    self.activation_counter = [] # array of tuple of layer name, output activations
    self.activation_counter.append(('input', mc.IMAGE_WIDTH*mc.IMAGE_HEIGHT*3))
    #self.is_training = tf.placeholder(tf.bool, [])
    
  def _add_act(self,act,name):
      self.act += [act]
      self.act_names += [name]

  def _add_debug(self,val,name):
    self.debug_val += [val]
    self.debug_val_names += [name]
    
  def _add_forward_graph(self):
    """NN architecture specification."""
    raise NotImplementedError

  def _add_interpretation_graph(self):
    """Interpret NN output."""
    mc = self.mc

    with tf.variable_scope('interpret_output') as scope:
      preds = self.preds

      print ("preds:",preds)
      #print ('mc.CLASSES:',mc.CLASSES)
      #input('')
      
      # probability
      num_class_probs = mc.ANCHOR_PER_GRID*mc.CLASSES
      print ("num_class_probs:",num_class_probs)
      
      self.pred_class_probs = tf.reshape(
          tf.nn.softmax(
              tf.reshape(
                  preds[:, :, :, :num_class_probs],
                  [-1, mc.CLASSES]
              )
          ),
          [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
          name='pred_class_probs'
      )
      print ("mc.ANCHORS:",mc.ANCHORS)
      
      print ("self.pred_class_probs:",self.pred_class_probs)
      #self._add_debug(self.pred_class_probs,"pred_class_probs")
      self._add_act(self.pred_class_probs,"pred_class_probs")
      
      # confidence
      num_confidence_scores = mc.ANCHOR_PER_GRID+num_class_probs
      self.pred_conf = tf.sigmoid(
          tf.reshape(
              preds[:, :, :, num_class_probs:num_confidence_scores],
              [mc.BATCH_SIZE, mc.ANCHORS]
          ),
          name='pred_confidence_score'
      )
      print ("num_confidence_scores:",num_confidence_scores)
      self._add_act(self.pred_conf,"pred_conf")
      
      # bbox_delta
      self.pred_box_delta = tf.reshape(
          preds[:, :, :, num_confidence_scores:],
          [mc.BATCH_SIZE, mc.ANCHORS, 4],
          name='bbox_delta'
      )
      self._add_act(self.pred_box_delta,"bbox_delta")
      
      # number of object. Used to normalize bbox and classification loss
      self.num_objects = tf.reduce_sum(self.input_mask, name='num_objects')

    with tf.variable_scope('bbox') as scope:
      with tf.variable_scope('stretching'):
        delta_x, delta_y, delta_w, delta_h = tf.unstack(
            self.pred_box_delta, axis=2)

        anchor_x = mc.ANCHOR_BOX[:, 0]
        anchor_y = mc.ANCHOR_BOX[:, 1]
        anchor_w = mc.ANCHOR_BOX[:, 2]
        anchor_h = mc.ANCHOR_BOX[:, 3]

        box_center_x = tf.identity(
            anchor_x + delta_x * anchor_w, name='bbox_cx')
        box_center_y = tf.identity(
            anchor_y + delta_y * anchor_h, name='bbox_cy')
        box_width = tf.identity(
            anchor_w * util.safe_exp(delta_w, mc.EXP_THRESH),
            name='bbox_width')
        box_height = tf.identity(
            anchor_h * util.safe_exp(delta_h, mc.EXP_THRESH),
            name='bbox_height')

        self._activation_summary(delta_x, 'delta_x')
        self._activation_summary(delta_y, 'delta_y')
        self._activation_summary(delta_w, 'delta_w')
        self._activation_summary(delta_h, 'delta_h')

        self._activation_summary(box_center_x, 'bbox_cx')
        self._activation_summary(box_center_y, 'bbox_cy')
        self._activation_summary(box_width, 'bbox_width')
        self._activation_summary(box_height, 'bbox_height')

      with tf.variable_scope('trimming'):
        xmins, ymins, xmaxs, ymaxs = util.bbox_transform(
            [box_center_x, box_center_y, box_width, box_height])

        # The max x position is mc.IMAGE_WIDTH - 1 since we use zero-based
        # pixels. Same for y.
        xmins = tf.minimum(
            tf.maximum(0.0, xmins), mc.IMAGE_WIDTH-1.0, name='bbox_xmin')
        self._activation_summary(xmins, 'box_xmin')

        self._add_act(xmins,"xmins")
        
        ymins = tf.minimum(
            tf.maximum(0.0, ymins), mc.IMAGE_HEIGHT-1.0, name='bbox_ymin')
        self._activation_summary(ymins, 'box_ymin')

        xmaxs = tf.maximum(
            tf.minimum(mc.IMAGE_WIDTH-1.0, xmaxs), 0.0, name='bbox_xmax')
        self._activation_summary(xmaxs, 'box_xmax')

        ymaxs = tf.maximum(
            tf.minimum(mc.IMAGE_HEIGHT-1.0, ymaxs), 0.0, name='bbox_ymax')
        self._activation_summary(ymaxs, 'box_ymax')
        self._add_act(ymaxs,"ymaxs")
        
        self.det_boxes = tf.transpose(
            tf.stack(util.bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])),
            (1, 2, 0), name='bbox'
        )
        
        self._add_act(self.det_boxes,"det_boxes")
        
    with tf.variable_scope('IOU'):
      def _tensor_iou(box1, box2):
        with tf.variable_scope('intersection'):
          xmin = tf.maximum(box1[0], box2[0], name='xmin')
          ymin = tf.maximum(box1[1], box2[1], name='ymin')
          xmax = tf.minimum(box1[2], box2[2], name='xmax')
          ymax = tf.minimum(box1[3], box2[3], name='ymax')

          w = tf.maximum(0.0, xmax-xmin, name='inter_w')
          h = tf.maximum(0.0, ymax-ymin, name='inter_h')
          intersection = tf.multiply(w, h, name='intersection')

        with tf.variable_scope('union'):
          w1 = tf.subtract(box1[2], box1[0], name='w1')
          h1 = tf.subtract(box1[3], box1[1], name='h1')
          w2 = tf.subtract(box2[2], box2[0], name='w2')
          h2 = tf.subtract(box2[3], box2[1], name='h2')

          union = w1*h1 + w2*h2 - intersection

        return intersection/(union+mc.EPSILON) \
            * tf.reshape(self.input_mask, [mc.BATCH_SIZE, mc.ANCHORS])
      
      self.ious = self.ious.assign(
          _tensor_iou(
              util.bbox_transform(tf.unstack(self.det_boxes, axis=2)),
              util.bbox_transform(tf.unstack(self.box_input, axis=2))
          )
      )
      #self._add_act(self.ious, 'ious')
      #print ("ious:",self.ious)
      
      self._activation_summary(self.ious, 'conf_score')

    with tf.variable_scope('probability') as scope:
      self._activation_summary(self.pred_class_probs, 'class_probs')

      probs = tf.multiply(
          self.pred_class_probs,
          tf.reshape(self.pred_conf, [mc.BATCH_SIZE, mc.ANCHORS, 1]),
          name='final_class_prob'
      )

      self._activation_summary(probs, 'final_class_prob')
      
      self.det_probs = tf.reduce_max(probs, 2, name='score')
      self._add_act(self.det_probs, 'det_probs')
      
      self.det_class = tf.argmax(probs, 2, name='class_idx')
      self._add_act(self.det_class, 'det_class')

  def _loss_v2(self,ious,pred,gamma=2.1):
    #rate = tf.abs(pred - ious) / (ious + 0.00000001)
    y = 1.5 * tf.abs(pred - ious)**gamma
    return y

  def _loss_v1(self,ious,pred):
    rate = tf.abs(pred - ious) / ious
    bool_1 = rate < 0.35
    y = tf.where(bool_1, tf.zeros_like(pred), pred - ious)
    return y
    
  def _sigmoid_loss(self,x):
    #test list
    #{8,0.43},{12,0.40},{24,0.36}
    #c = tf.constant([0.15],shape=[12,16848])
    
    #a = tf.Variable(tf.constant(0., shape=[12,16848]), name='a', trainable=False)
    #return tf.cond(, lambda: x-x, lambda: x)
    mc = self.mc
    return tf.clip_by_value(1.0/(1.0+tf.exp(-mc.CONF_LOSS_GAMMA*(x+mc.CONF_LOSS_X_OFFSET))) 
                    + mc.CONF_LOSS_Y_OFFSET,0,1)
    
    
  def _add_loss_graph(self):
    """Define the loss operation."""
    mc = self.mc

    with tf.variable_scope('class_regression') as scope:
      # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
      # add a small value into log to prevent blowing up
      #  mc.B_FOCAL_LOSS          = True 
      # mc.FOCAL_LOSS_APHA       = 0.25
      # mc.FOCAL_LOSS_GAMMA      = 2 
      
      if mc.B_FOCAL_LOSS == True:
        alpha_t = self.input_mask * mc.FOCAL_LOSS_APHA + (1 - self.input_mask) * (1 - mc.FOCAL_LOSS_APHA)
        pt = self.labels*self.pred_class_probs + (1-self.labels)*(1-self.pred_class_probs)
        self.class_loss = tf.truediv(
            tf.reduce_sum(-alpha_t * ((1 - pt + mc.EPSILON)**mc.FOCAL_LOSS_GAMMA)  * tf.log(pt+mc.EPSILON)
                * mc.LOSS_COEF_CLASS),
            float(mc.ANCHORS),
            name='class_loss'
        )
        
      else:
        self.class_loss = tf.truediv(
            tf.reduce_sum(
                (self.labels*(-tf.log(self.pred_class_probs+mc.EPSILON))
                 + (1-self.labels)*(-tf.log(1-self.pred_class_probs+mc.EPSILON)))
                * self.input_mask * mc.LOSS_COEF_CLASS),
            self.num_objects,
            name='class_loss'
        )
      tf.add_to_collection('losses', self.class_loss)

    if mc.B_CONF_LOSS_V1 == True:
        with tf.variable_scope('confidence_score_regression') as scope:
            input_mask = tf.reshape(self.input_mask, [mc.BATCH_SIZE, mc.ANCHORS])
            self.conf_loss = tf.reduce_mean(
              tf.reduce_sum(
                  tf.square((self.ious - self.pred_conf)) 
                  * input_mask*mc.LOSS_COEF_CONF_POS/self.num_objects
                     +self._sigmoid_loss(tf.square((self.ious - self.pred_conf)))
                  *(1-input_mask)*mc.LOSS_COEF_CONF_NEG/(mc.ANCHORS-self.num_objects),
                  reduction_indices=[1]
              ),
              name='confidence_loss'
            )
            tf.add_to_collection('losses', self.conf_loss)
            tf.summary.scalar('mean iou', tf.reduce_sum(self.ious)/self.num_objects)
            
            self.conf_loss_positive = tf.reduce_mean(
              tf.reduce_sum(
                  tf.square((self.ious - self.pred_conf))
                  * (input_mask*mc.LOSS_COEF_CONF_POS/self.num_objects),
                  reduction_indices=[1]
              ),
              name='confidence_loss_positive'
            )
            self.conf_loss_negtive = tf.reduce_mean(
              tf.reduce_sum(
                  self._sigmoid_loss(tf.square((self.ious - self.pred_conf)))
                  * (1-input_mask)*mc.LOSS_COEF_CONF_NEG/(mc.ANCHORS-self.num_objects),
                  reduction_indices=[1]
              ),
              name='confidence_loss_negitive'
            )
    else:
        with tf.variable_scope('confidence_score_regression') as scope:
          input_mask = tf.reshape(self.input_mask, [mc.BATCH_SIZE, mc.ANCHORS])
          self.conf_loss = tf.reduce_mean(
              tf.reduce_sum(
                  tf.square((self.ious - self.pred_conf)) 
                  * (input_mask*mc.LOSS_COEF_CONF_POS/self.num_objects
                     +(1-input_mask)*mc.LOSS_COEF_CONF_NEG/(mc.ANCHORS-self.num_objects)),
                  reduction_indices=[1]
              ),
              name='confidence_loss'
          )
          tf.add_to_collection('losses', self.conf_loss)
          tf.summary.scalar('mean iou', tf.reduce_sum(self.ious)/self.num_objects)

        self.conf_loss_positive = tf.reduce_mean(
              tf.reduce_sum(
                  tf.square((self.ious - self.pred_conf)) 
                  * (input_mask*mc.LOSS_COEF_CONF_POS/self.num_objects),
                  reduction_indices=[1]
              ),
              name='confidence_loss_positive'
          )
        self.conf_loss_negtive = tf.reduce_mean(
              tf.reduce_sum(
                  tf.square((self.ious - self.pred_conf)) 
                  * (1-input_mask)*mc.LOSS_COEF_CONF_NEG/(mc.ANCHORS-self.num_objects),
                  reduction_indices=[1]
              ),
              name='confidence_loss_negitive'
        )
        
        
    with tf.variable_scope('bounding_box_regression') as scope:
      self.bbox_loss = tf.truediv(
          tf.reduce_sum(
              mc.LOSS_COEF_BBOX * tf.square(
                  self.input_mask*(self.pred_box_delta-self.box_delta_input))),
          self.num_objects,
          name='bbox_loss'
      )
      tf.add_to_collection('losses', self.bbox_loss)

    # add above losses as well as weight decay losses to form the total loss
    self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

  def _add_weight_decay_to_loss(self,weight_decay,weights_name='weights'):
    trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = [v for v in trainable if weights_name in v.name]

    print ('================>len(kernels):',len(kernels))
    print ('kernels:',kernels)
    
    for K in kernels:  
        l2_loss = tf.multiply(
            weight_decay, tf.nn.l2_loss(K), name='l2_loss'
        )
        tf.add_to_collection('losses', l2_loss)
        

  def _add_train_graph_v1(self,weights_name='weights'):
    mc = self.mc
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                                  self.global_step,
                                  mc.DECAY_STEPS,
                                  mc.LR_DECAY_FACTOR,
                                  staircase=True)
    tf.summary.scalar('learning_rate', self.lr)
    
    optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=mc.MOMENTUM)

    if mc.ADD_WEIGHT_DECAY_TO_LOSS == True:
      if mc.VERSION == 'V0':
        self._add_weight_decay_to_loss(mc.WEIGHT_DECAY,weights_name)

    #_add_loss_summaries(self.loss)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
      grads_vars = optimizer.compute_gradients(self.loss,tf.trainable_variables())
      with tf.variable_scope('clip_gradient') as scope:
        for i, (grad, var) in enumerate(grads_vars):
          print ("2--var.name:",var.op.name)
          grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)
          
      apply_gradient_op = optimizer.apply_gradients(grads_vars,global_step=self.global_step)
      with tf.control_dependencies([apply_gradient_op]):
        self.train_op = tf.no_op(name='train')
        
       
  def _add_train_graph(self):
    """Define the training operation."""
    mc = self.mc
    
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                                    self.global_step,
                                    mc.DECAY_STEPS,
                                    mc.LR_DECAY_FACTOR,
                                    staircase=True)

    tf.summary.scalar('learning_rate', self.lr)

    _add_loss_summaries(self.loss)

    opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=mc.MOMENTUM)
    grads_vars = opt.compute_gradients(self.loss, tf.trainable_variables())

    #这里在把保存的weights还原回来
    for var in tf.global_variables():
        print ("1--var.name:",var.op.name)

    
    with tf.variable_scope('clip_gradient') as scope:
      for i, (grad, var) in enumerate(grads_vars):
        print ("2--var.name:",var.op.name)
        grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)
        
   # for var in tf.global_variables():
    #    print ("11--var.name:",var.op.name)
    apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.global_step)    
    for var in tf.trainable_variables():
        print ("3--var.name:",var.op.name)
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads_vars:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    with tf.control_dependencies([apply_gradient_op]):
      self.train_op = tf.no_op(name='train')
    
  def _add_viz_graph(self):
    """Define the visualization operation."""
    mc = self.mc
    self.image_to_show = tf.placeholder(
        tf.float32, [None, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],
        name='image_to_show'
    )
    self.viz_op = tf.summary.image('sample_detection_results',
        self.image_to_show, collections='image_summary',
        max_outputs=mc.BATCH_SIZE)

  def _batch_norm_layer(self,x, scope_bn):
    mc = self.mc
        
    with tf.variable_scope(scope_bn):
                                  
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        axises = list(range(len(x.shape) - 1))
        print ('===>axises:',axises)
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=mc.BATH_NORM_DECAY)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        if mc.is_training == True:
          mean, var = mean_var_with_update()
        else:
          mean, var = ema.average(batch_mean), ema.average(batch_var)
        '''
        mean, var = tf.cond(mc.is_training, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        '''
        
        normed = tf.nn.batch_normalization(x, mean, float(var), beta, gamma, mc.BATCH_NORM_EPSILON)
        #normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, mc.BATCH_NORM_EPSILON)
    return  normed 
  
  

  def _conv_bn_layer(
      self, layer_name, inputs, filters, size, stride, padding='SAME',use_bias=False,
      freeze=False, xavier=False, relu=True, activation_fn=tf.nn.relu,stddev=0.001,
                kernel_name='kernels',bias_name='biases'):
    """Convolutional layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """

    mc = self.mc
    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
        bias_val = cw[layer_name][1]
        # check the shape
        if (kernel_val.shape == 
              (size, size, inputs.get_shape().as_list()[-1], filters)) \
           and (bias_val.shape == (filters, )):
          use_pretrained_param = True
        else:
          print ('Shape of the pretrained parameter of {} does not match, '
              'use randomly initialized parameter'.format(layer_name))
      else:
        print ('Cannot find {} in the pretrained model. Use randomly initialized '
               'parameters'.format(layer_name))

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      channels = inputs.get_shape()[3]

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val , dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.0)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(0.0)
        
      kernel = _variable_with_weight_decay(
          kernel_name, shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

     #kernel_binary = binarize(kernel)
      if use_bias == True:
        biases = _variable_on_device(bias_name, [filters], bias_init, 
                                trainable=(not freeze))
        self.model_params += [kernel, biases]

      if mc.bDoreFa == True:
          fw, fa, fg = get_dorefa(mc.BITW,mc.BITA,mc.BITG)
          kernel = fw(kernel)
          if mc.BITA != 32:
            #inputs = tf.clip_by_value(inputs, 0.0, 1.0)
            inputs = inputs / tf.reduce_max(inputs) 
            inputs = fa(inputs)
            
      if mc.bQuant == True:
        if mc.bQuantWeights == True:
          kernel = self._quant_kernel_v1(mc,kernel)
      
      if mc.bQuant == True:
        if mc.bQuantActivations == True:
          inputs = self._quant_activations(mc,inputs)
            
      conv = tf.nn.conv2d(
          inputs, kernel, [1, stride, stride, 1], padding=padding,
          name='convolution')
      
      if use_bias == True:
        out0 = tf.nn.bias_add(conv, biases, name='bias_add')
      else:
        out0 = conv

      out0 = slim.batch_norm(out0, scope='BatchNorm')        
      if relu == True:
        out = activation_fn(out0, 'relu')
      else:
        out = out0

      self.model_size_counter.append(
          (layer_name, (1+size*size*int(channels))*filters)
      )
      out_shape = out.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append(
          (layer_name, out_shape[1]*out_shape[2]*out_shape[3])
      )
      
      return out

  def l2_normalization(self,
          layername,
          inputs,
          init_var=20.,
          reuse=None,
          trainable=True,
          scope=None):
      """Implement L2 normalization on every feature (i.e. spatial normalization).

      Should be extended in some near future to other dimensions, providing a more
      flexible normalization framework.

      Args:
        inputs: a 4-D tensor with dimensions [batch_size, height, width, channels].
        scaling: whether or not to add a post scaling operation along the dimensions
          which have been normalized.
        scale_initializer: An initializer for the weights.
        reuse: whether or not the layer and its variables should be reused. To be
          able to reuse the layer scope must be given.
        variables_collections: optional list of collections for all the variables or
          a dictionary containing a different list of collection per variable.
        outputs_collections: collection to add the outputs.
        data_format:  NHWC or NCHW data format.
        trainable: If `True` also add variables to the graph collection
          `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        scope: Optional scope for `variable_scope`.
      Returns:
        A `Tensor` representing the output of the operation.
      """
      mc = self.mc
      
      sacle_init_value = None
      if mc.LOAD_PRETRAINED_MODEL:
        cw = self.caffemodel_weight
        if layername in cw:
          sacle_init_value = np.array(cw[layername])
          
      with variable_scope.variable_scope(
              scope, 'L2Normalization', [inputs], reuse=reuse) as sc:
          inputs_shape = inputs.get_shape()
          inputs_rank = inputs_shape.ndims
          dtype = inputs.dtype.base_dtype
          
          # norm_dim = tf.range(1, inputs_rank-1)
          norm_dim = tf.range(inputs_rank-1, inputs_rank)
          params_shape = inputs_shape[-1:]

          # Normalize along spatial dimensions.
          outputs = nn.l2_normalize(inputs, norm_dim, epsilon=1e-12)

          #print ("l2_normalization params_shape:",params_shape)
          #raw_input('Enter and continue')
          
          # Additional scaling.  
          if sacle_init_value == None:
            scale_initializer = tf.constant(init_var,shape=params_shape,dtype=tf.float32)
          else:
            scale_initializer = tf.constant(sacle_init_value,shape=params_shape,dtype=tf.float32)
            
          scale_var = tf.get_variable('scale', initializer=scale_initializer, trainable=trainable)     
          return tf.multiply(outputs, scale_var)

  def _conv_layer(
      self, layer_name, inputs, filters, size, stride, padding='SAME',use_bias=True,
      freeze=False, xavier=False, relu=True, stddev=0.001,dilation=-1,
          kernel_name='kernels',bias_name='biases'):
    """Convolutional layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """

    mc = self.mc
    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
        bias_val = cw[layer_name][1]
        # check the shape
        if (kernel_val.shape == 
              (size, size, inputs.get_shape().as_list()[-1], filters)) \
           and (bias_val.shape == (filters, )):
          use_pretrained_param = True
          print ('load {} finish kernel_val.shape:{}'.format(layer_name,kernel_val.shape))
        else:
          print ('Shape of the pretrained parameter of {} does not match, '
              'use randomly initialized parameter'.format(layer_name))
      else:
        print ('Cannot find {} in the pretrained model. Use randomly initialized '
               'parameters'.format(layer_name))

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      channels = inputs.get_shape()[3]

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val , dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.0)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(0.0)
        
      kernel = _variable_with_weight_decay(
          kernel_name, shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))
      self.model_params +=  [kernel]
      
     #kernel_binary = binarize(kernel)
      if use_bias == True:
        biases = _variable_on_device(bias_name, [filters], bias_init, 
                                trainable=(not freeze))
        self.model_params += [biases]

      if mc.bQuant == True:
        if mc.bQuantWeights == True:
          kernel = self._quant_kernel_v1(mc,kernel)

      if mc.bQuant == True:
        if mc.bQuantActivations == True:
          inputs = self._quant_activations(mc,inputs)

      if dilation > 0:
        '''
        conv = tf.nn.convolution(inputs,kernel,padding=padding,
            strides=[1, stride, stride, 1],dilation_rate=dilation,name='convolution')
        '''
        conv = tf.nn.atrous_conv2d(
            inputs, kernel,padding=padding,rate=dilation,
            name='convolution')
        
      else:
        conv = tf.nn.conv2d(
            inputs, kernel, [1, stride, stride, 1], padding=padding,
            name='convolution')  

      if use_bias == True:
        out0 = tf.nn.bias_add(conv, biases, name='bias_add')
      else:
        out0 = conv
        
      if relu:
        out = tf.nn.relu(out0, 'relu')
      else:
        out = out0

      self.model_size_counter.append(
          (layer_name, (1+size*size*int(channels))*filters)
      )
      out_shape = out.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append(
          (layer_name, out_shape[1]*out_shape[2]*out_shape[3])
      )
      
      return out
  
  def _pooling_layer(
      self, layer_name, inputs, size, stride, pool_type='MAX',padding='SAME'):
    """Pooling layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
    Returns:
      A pooling layer operation.
    """

    with tf.variable_scope(layer_name) as scope:
      if pool_type == 'AVERAGE':
        out =  tf.nn.avg_pool(inputs, 
                              ksize=[1, size, size, 1], 
                              strides=[1, stride, stride, 1],
                              padding=padding)
      else:
        out =  tf.nn.max_pool(inputs, 
                              ksize=[1, size, size, 1], 
                              strides=[1, stride, stride, 1],
                              padding=padding)
                              
      activation_size = np.prod(out.get_shape().as_list()[1:])
      self.activation_counter.append((layer_name, activation_size))
      return out

  
  def _fc_layer(
      self, layer_name, inputs, hiddens, flatten=False, relu=True,
      xavier=False, stddev=0.001):
    """Fully connected layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      hiddens: number of (hidden) neurons in this layer.
      flatten: if true, reshape the input 4D tensor of shape 
          (batch, height, weight, channel) into a 2D tensor with shape 
          (batch, -1). This is used when the input to the fully connected layer
          is output of a convolutional layer.
      relu: whether to use relu or not.
      xavier: whether to use xavier weight initializer or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A fully connected layer operation.
    """
    mc = self.mc

    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        use_pretrained_param = True
        kernel_val = cw[layer_name][0]
        bias_val = cw[layer_name][1]

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      input_shape = inputs.get_shape().as_list()
      if flatten:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs = tf.reshape(inputs, [-1, dim])
        if use_pretrained_param:
          try:
            # check the size before layout transform
            assert kernel_val.shape == (hiddens, dim), \
                'kernel shape error at {}'.format(layer_name)
            kernel_val = np.reshape(
                np.transpose(
                    np.reshape(
                        kernel_val, # O x (C*H*W)
                        (hiddens, input_shape[3], input_shape[1], input_shape[2])
                    ), # O x C x H x W
                    (2, 3, 1, 0)
                ), # H x W x C x O
                (dim, -1)
            ) # (H*W*C) x O
            # check the size after layout transform
            assert kernel_val.shape == (dim, hiddens), \
                'kernel shape error at {}'.format(layer_name)
          except:
            # Do not use pretrained parameter if shape doesn't match
            use_pretrained_param = False
            print ('Shape of the pretrained parameter of {} does not match, '
                   'use randomly initialized parameter'.format(layer_name))
      else:
        dim = input_shape[1]
        if use_pretrained_param:
          try:
            kernel_val = np.transpose(kernel_val, (1,0))
            assert kernel_val.shape == (dim, hiddens), \
                'kernel shape error at {}'.format(layer_name)
          except:
            use_pretrained_param = False
            print ('Shape of the pretrained parameter of {} does not match, '
                   'use randomly initialized parameter'.format(layer_name))

      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val, dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.0)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(0.0)

      weights = _variable_with_weight_decay(
          'weights', shape=[dim, hiddens], wd=mc.WEIGHT_DECAY,
          initializer=kernel_init)
      biases = _variable_on_device('biases', [hiddens], bias_init)
      self.model_params += [weights, biases]
  
      outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
      if relu:
        outputs = tf.nn.relu(outputs, 'relu')

      # count layer stats
      self.model_size_counter.append((layer_name, (dim+1)*hiddens))

      num_flops = 2 * dim * hiddens + hiddens
      if relu:
        num_flops += 2*hiddens
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append((layer_name, hiddens))

      return outputs

  def filter_prediction(self, boxes, probs, cls_idx,backgroud_id=-1):
    """Filter bounding box predictions with probability threshold and
    non-maximum supression.

    Args:
      boxes: array of [cx, cy, w, h].
      probs: array of probabilities
      cls_idx: array of class indices
    Returns:
      final_boxes: array of filtered bounding boxes.
      final_probs: array of filtered probabilities
      final_cls_idx: array of filtered class indices
    """

    mc = self.mc

    '''
    if backgroud_id >= 0:
      print ('remove backgroud')
      order_forcegroud = np.where(cls_idx != backgroud_id)
      probs = probs[order_forcegroud]
      boxes = boxes[order_forcegroud]
      cls_idx = cls_idx[order_forcegroud]
    '''
    
    if mc.TOP_N_DETECTION < len(probs) and mc.TOP_N_DETECTION > 0:
      #print ('[filter_prediction]============1')
      order = probs.argsort()[:-mc.TOP_N_DETECTION-1:-1]
      probs = probs[order]
      boxes = boxes[order]
      cls_idx = cls_idx[order]
    else:
      filtered_idx = np.nonzero(probs>mc.PROB_THRESH)[0]
      probs = probs[filtered_idx]
      boxes = boxes[filtered_idx]
      cls_idx = cls_idx[filtered_idx]
      
    final_boxes = []
    final_probs = []
    final_cls_idx = []
    #print ('probs:',probs)
    #print ('3===========cls_idx.shape:',cls_idx.shape)
    
    for c in range(mc.CLASSES):
      if backgroud_id >= 0:
        if c == backgroud_id:
          continue
          
      idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
      
      keep = util.nms(boxes[idx_per_class], probs[idx_per_class], mc.NMS_THRESH)
      #print ("c",c," keep:",keep)
       
      for i in range(len(keep)):
        if keep[i]:
          final_boxes.append(boxes[idx_per_class[i]])
          final_probs.append(probs[idx_per_class[i]])
          final_cls_idx.append(c)
    return final_boxes, final_probs, final_cls_idx

  def _activation_summary(self, x, layer_name):
    """Helper to create summaries for activations.

    Args:
      x: layer output tensor
      layer_name: name of the layer
    Returns:
      nothing
    """
    with tf.variable_scope('activation_summary') as scope:
      tf.summary.histogram(
          'activation_summary/'+layer_name, x)
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/sparsity', tf.nn.zero_fraction(x))
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/average', tf.reduce_mean(x))
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/max', tf.reduce_max(x))
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/min', tf.reduce_min(x))
