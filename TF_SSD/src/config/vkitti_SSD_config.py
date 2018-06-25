
"""Model configuration for pascal dataset"""

import numpy as np
import math
from config import base_model_config
from easydict import EasyDict as edict

def vkitti_SSD_config():
  """Specify the parameters to tune below."""
  
  mc                       = base_model_config('VKITTI')
  mc.VERSION               = 'V1'
  mc.IMAGE_WIDTH           = 300
  mc.IMAGE_HEIGHT          = 300
  mc.BGR_MEANS             = np.array([[[104,117,123]]])
  mc.BATCH_SIZE            = 20

  mc.WEIGHT_DECAY          = 0.0005 #0.00001
  mc.LEARNING_RATE         = 0.01
  mc.DECAY_STEPS           = 20000
  mc.LR_DECAY_FACTOR       = 0.5
  mc.MAX_GRAD_NORM         = 1.0 #1.0
  mc.MOMENTUM              = 0.9
  
  mc.mbox_source_layers_name = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
  mc.normalizations = [20, -1, -1, -1, -1, -1]
  mc.bQuant         = False
  cal_prior_param(mc)

  mc.BACKGROUD_ID         = 0
  mc.TOP_N_DETECTION      = 400
  mc.PROB_THRESH          = 0.01
  mc.NMS_THRESH           = 0.45
  mc.PLOT_PROB_THRESH     = 0.4
  mc.overlap_threshold    = 0.5
  #mc.keep_top_k           = 200
  print ('vkitti_SSD_config')


  #data layer params
  mc.batch_sampler  = get_batch_sampler()
  mc.expand_param = edict()
  mc.expand_param.prob = 1.0
  mc.expand_param.min_expand_ratio = 1.0
  mc.expand_param.max_expand_ratio = 3.0

  #multibox_loss layer params
  mc.multibox_loss_param = edict()
  mc.multibox_loss_param.overlap_threshold = 0.5
  mc.multibox_loss_param.neg_pos_ratio = 3.
  mc.multibox_loss_param.neg_overlap = 0.5
  mc.background_label_id = 0

  #anchors
  mc.ANCHOR_BOX,mc.ANCHORS_NUM,mc.ANCHOR_PER_GRID  = all_anchors(mc) 
  mc.PRIORBOX_VARIANCES  = all_prior_variance(mc)

  '''
  print ('np.shape(mc.ANCHOR_BOX):',np.shape(mc.ANCHOR_BOX))
  print ('mc.ANCHORS_NUM:',mc.ANCHORS_NUM)
  print ('np.shape(mc.PRIORBOX_VARIANCES):',np.shape(mc.PRIORBOX_VARIANCES))
  priorboxes = np.concatenate([mc.ANCHOR_BOX,mc.PRIORBOX_VARIANCES],axis=0)
  save_array_to_txt_file('all_anchors.txt',mc.ANCHOR_BOX)
  save_array_to_txt_file('all_prior_variance.txt',mc.PRIORBOX_VARIANCES)
  save_array_to_txt_file('_priorboxes.txt',priorboxes)
  raw_input('pause')
  '''

  mc.ADD_WEIGHT_DECAY_TO_LOSS = True

  mc.NUM_THREAD            = 4
  mc.QUEUE_CAPACITY        = 100
  return mc


def save_array_to_txt_file(filename,arr): 
    arr_flat = arr.reshape(arr.size)
    filename = "debug_txt/" + filename
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
    
def cal_prior_param(mc):

  min_dim = mc.IMAGE_HEIGHT
  min_ratio = 20
  max_ratio = 90
  step = int(math.floor((max_ratio - min_ratio) / (len(mc.mbox_source_layers_name) - 2)))
  mc.min_sizes = []
  mc.max_sizes = []
  
  for ratio in xrange(min_ratio, max_ratio + 1, step):
    mc.min_sizes.append(min_dim * ratio / 100.)
    mc.max_sizes.append(min_dim * (ratio + step) / 100.)
  mc.min_sizes = [min_dim * 10 / 100.] + mc.min_sizes
  mc.max_sizes = [min_dim * 20 / 100.] + mc.max_sizes
  print ('min_sizes:',mc.min_sizes)
  print ('max_sizes:',mc.max_sizes)
  
  mc.steps = [8, 16, 32, 64, 100, 300]
  mc.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
  mc.prior_variance = [0.1, 0.1, 0.2, 0.2] 
  mc.box_offset = 0.5
  mc.prior_layers_space_shape = [[38,38],[19,19],[10,10],[5,5],[3,3],[1,1]]


def get_num_priors(mc,src_idx):
  min_size = mc.min_sizes[src_idx]
  max_size = mc.max_sizes[src_idx]
    
  num_priors_ = 0
  num_priors_ += 1

  for var in mc.aspect_ratios[src_idx]:
    num_priors_ += 2

  if max_size:
    num_priors_ += 1

  return num_priors_
  
def all_prior_variance(mc):
  prior_variances = []
  all_anchors_num = 0
  for i, space in enumerate(mc.prior_layers_space_shape):
    variance = np.array(mc.prior_variance)  
    num_priors = get_num_priors(mc,i)
    prior_variance = np.reshape([variance]*space[0] * space[1] * num_priors,-1) 
    prior_variances.append(prior_variance)
    all_anchors_num += num_priors * space[0] * space[1]
    
  return np.reshape(np.concatenate(prior_variances,axis=0),(all_anchors_num,-1))
  
def all_anchors(mc):
  anchors = []
  all_anchor_per_grid = 0
  all_anchors_num = 0
  for i, space in enumerate(mc.prior_layers_space_shape):
    anchor_per_grid = get_num_priors(mc,i)
    top_data = Anchors(mc,i)
    anchors.append(top_data)
    #all_anchor_per_grid +=  anchor_per_grid
    all_anchors_num += anchor_per_grid * space[0] * space[1]

  return  np.concatenate(anchors,axis=0),all_anchors_num,all_anchor_per_grid
    
def Anchors(mc,src_idx):
    min_size = mc.min_sizes[src_idx]
    max_size = mc.max_sizes[src_idx]
    aspect_ratios = []
    aspect_ratios.append(1.)
    for var in mc.aspect_ratios[src_idx]:
      aspect_ratios.append(var)
      aspect_ratios.append(1./var)
    
    num_priors_ = get_num_priors(mc,src_idx)
    step_w = step_h = mc.steps[src_idx]

    layer_height = mc.prior_layers_space_shape[src_idx][0]
    layer_width = mc.prior_layers_space_shape[src_idx][1]
    offset_ = mc.box_offset
    top_data = np.zeros((layer_width * layer_height * num_priors_,4))

    idx = 0
    for h in range(0,layer_height):
      for w in range(0,layer_width):
        center_x = (w + offset_) * step_w
        center_y = (h + offset_) * step_h
        box_width = box_height = min_size
        top_data[idx][0] = (center_x - box_width / 2.) / mc.IMAGE_WIDTH
        top_data[idx][1] = (center_y - box_height / 2.) / mc.IMAGE_HEIGHT
        top_data[idx][2] = (center_x + box_width / 2.) / mc.IMAGE_WIDTH
        top_data[idx][3] = (center_y + box_height / 2.) / mc.IMAGE_HEIGHT
        idx += 1
 
        box_width = box_height = math.sqrt(min_size * max_size)
        top_data[idx][0] = (center_x - box_width / 2.) / mc.IMAGE_WIDTH
        top_data[idx][1] = (center_y - box_height / 2.) / mc.IMAGE_HEIGHT
        top_data[idx][2] = (center_x + box_width / 2.) / mc.IMAGE_WIDTH
        top_data[idx][3] = (center_y + box_height / 2.) / mc.IMAGE_HEIGHT
        idx += 1

        for i in range(0,len(aspect_ratios)):
          ar = aspect_ratios[i]
          if math.fabs(ar - 1.) < 1e-6:
            continue
          box_width = min_size * math.sqrt(ar)
          box_height = min_size / math.sqrt(ar)
          top_data[idx][0] = (center_x - box_width / 2.) / mc.IMAGE_WIDTH
          top_data[idx][1] = (center_y - box_height / 2.) / mc.IMAGE_HEIGHT
          top_data[idx][2] = (center_x + box_width / 2.) / mc.IMAGE_WIDTH
          top_data[idx][3] = (center_y + box_height / 2.) / mc.IMAGE_HEIGHT
          idx += 1
    
    return top_data

    
def add_default_sampler(batch_sampler):
    keys = ['min_scale','max_scale','min_aspect_ratio','max_aspect_ratio',
        'min_jaccard_overlap','max_jaccard_overlap','max_trials','max_sample']
    for i in range(1,len(batch_sampler)):
        for key in keys:
            if batch_sampler[i].get(key) == None:
                default_val = batch_sampler[0][key]
                batch_sampler[i][key] = default_val

def get_batch_sampler():
     # the first item is the default values
    batch_sampler = [
        {
            'max_sample':1,
            'max_trials':1,
            'min_scale':1,
            'max_scale':1,
            'min_aspect_ratio':1,
            'max_aspect_ratio':1,
            'min_jaccard_overlap':0,
            'max_jaccard_overlap':1,
        },
        {
            'max_trials': 1,
            'max_sample': 1,
        },
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
            'min_jaccard_overlap': 0.1,
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
            'min_jaccard_overlap': 0.3,
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
            'min_jaccard_overlap': 0.5,
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
            'min_jaccard_overlap': 0.7,
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
            'min_jaccard_overlap': 0.9,
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
            'max_jaccard_overlap': 1.0,
            'max_trials': 50,
            'max_sample': 1,
        },
    ]
    
    add_default_sampler(batch_sampler)
    
    return batch_sampler

  
