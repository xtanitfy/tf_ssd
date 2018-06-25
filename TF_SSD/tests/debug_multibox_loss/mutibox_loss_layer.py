import tensorflow as tf


class MutiBoxLossLayer():
    def __init__(self, mc):
        self.mc = mc
        self.debug_val = []
        self.debug_val_names = []
        
    def _add_debug(self,var,name):
        self.debug_val.append(var)
        self.debug_val_names.append(name)
    
    
    def encode_bbox(self,input_mask_box,gt_boxes,prior_boxes,prior_variances):
          
        prior_width = prior_boxes[:,2] - prior_boxes[:,0]
        prior_height = prior_boxes[:,3] - prior_boxes[:,1]
        prior_center_x = (prior_boxes[:,0] + prior_boxes[:,2])/2.
        prior_center_y = (prior_boxes[:,1] + prior_boxes[:,3])/2.
        
        bbox_width = gt_boxes[:,:,2] - gt_boxes[:,:,0]
        bbox_height = gt_boxes[:,:,3] - gt_boxes[:,:,1]
        bbox_center_x = (gt_boxes[:,:,0] + gt_boxes[:,:,2])/2.
        bbox_center_y = (gt_boxes[:,:,1] + gt_boxes[:,:,3])/2.
        
        xmins = (bbox_center_x - prior_center_x) / prior_width / prior_variances[:,0]
        ymins = (bbox_center_y - prior_center_y) / prior_height / prior_variances[:,1]
        xmaxs = tf.log(bbox_width / prior_width) / prior_variances[:,2]
        ymaxs = tf.log(bbox_height / prior_height) / prior_variances[:,3]
        gt_encode_boxes = tf.stack([xmins, ymins, xmaxs, ymaxs],axis=2)
        
        self.gt_encode_boxes = tf.where(tf.equal(input_mask_box,1),gt_encode_boxes,tf.zeros_like(gt_encode_boxes))
        
    def cal_loc_loss(self,loc_data,prior_boxes,prior_variances,gt_boxes,input_mask):
        
        mc = self.mc
        masks = []
        for i in range(0,4):
            masks.append(tf.reshape(self.input_mask,[mc.BATCH_SIZE,-1,1]))
        input_mask_box = tf.concat(masks,2)
        
        match_loc_data = tf.where(tf.equal(input_mask_box,1),loc_data,tf.zeros_like(loc_data))
        gt_boxes = tf.where(tf.equal(input_mask_box,1),gt_boxes,tf.zeros_like(gt_boxes))
        
        self.encode_bbox(input_mask_box,gt_boxes,prior_boxes,prior_variances)   
        diff_abs = tf.abs(self.gt_encode_boxes - match_loc_data)
        loc_loss = tf.where(diff_abs < 1,0.5 * tf.square(diff_abs),(diff_abs - 0.5))
        self.loc_loss = tf.reduce_sum(loc_loss) / tf.reduce_sum(self.num_pos)
    
    def cal_conf_loss_v1(self):
        mc = self.mc
        self.pos_loss = tf.reduce_sum(tf.where(tf.equal(self.input_mask,1),self.conf_loss,tf.zeros_like(self.conf_loss)))  
        
        neg_losses = []
        for i in range(0,mc.BATCH_SIZE):
            neg_loss = tf.reduce_sum(self.batch_neg_loss[i])
            neg_losses.append([neg_loss])
        self.neg_loss = tf.reduce_sum(tf.concat(neg_losses,0)) 
        
        self.conf_loss = tf.add(self.pos_loss,self.neg_loss) / tf.reduce_sum(self.num_pos)
        
    def cal_conf_loss(self):
        mc = self.mc
        self.pos_loss = tf.reduce_sum(tf.where(tf.equal(self.input_mask,1),self.conf_loss,tf.zeros_like(self.conf_loss)))
        
        neg_losses = []
        for i in range(0,mc.BATCH_SIZE): 
            idx_reshape = tf.reshape(self.batch_neg_idx[i],[-1]) 
            print ('idx_reshape:',idx_reshape)
            self.mask = tf.sparse_to_dense(idx_reshape,[mc.NUM_PRIORBOX],True,False)
            neg_loss = tf.where(mask,self.conf_loss[i],tf.zeros_like(self.conf_loss[i]))
            #neg_losses.append(tf.reshape(neg_loss,[1,mc.NUM_PRIORBOX]))
            #neg_loss = tf.gather(self.conf_loss[i],self.batch_neg_idx[i])
            #neg_loss = tf.where(self.batch_neg_loss[i] == self.conf_loss[i],self.conf_loss[i],tf.zeros_like(self.conf_loss[i])) 
            #self.neg_loss = tf.reduce_sum(neg_loss)
                    
        #self.neg_loss = tf.concat(neg_losses,0)
        
        #print ('neg_loss_concat:',self.neg_loss)
        #self.conf_loss = tf.concat([neg_loss_concat,self.pos_loss],0)
        
        #self.conf_loss = (tf.reduce_sum(self.conf_loss) + tf.reduce_sum(self.neg_loss)) / tf.reduce_sum(self.num_pos) 
        #self.conf_loss = self.pos_loss
        

    def mine_hard_examples(self):
        mc = self.mc
        self.batch_neg_loss = []
        self.batch_neg_idx = []
        for i in range(0,mc.BATCH_SIZE):
            filter_neg = tf.where(tf.logical_and(tf.equal(self.input_mask[i],0),tf.less(self.all_match_overlaps[i],mc.neg_overlap)),
                                                self.conf_loss[i],tf.zeros_like(self.conf_loss[i]))
            neg_loss,neg_idx = tf.nn.top_k(filter_neg, k=self.num_neg[i], sorted=False)
            self.batch_neg_loss.append(neg_loss)
            self.batch_neg_idx.append(neg_idx)
        
    def process(self,loc_data,conf_data,prior_boxes,prior_variances,gt_boxes,gt_label,input_mask,all_match_overlaps):
        mc = self.mc
        
        self._add_debug(loc_data,"loc_data")
        self._add_debug(conf_data,"conf_data")
        
        self.input_mask = input_mask
        self.all_match_overlaps = all_match_overlaps
        
        conf_data_norm = tf.nn.softmax(
              tf.reshape(
                  conf_data,[-1, mc.NUM_CLASSES]
              )
          )
        conf_data_norm = tf.reshape(conf_data_norm,[mc.BATCH_SIZE, mc.NUM_PRIORBOX,mc.NUM_CLASSES])
        self.num_pos = tf.reduce_sum(input_mask,axis=1)
        self.num_neg = tf.cast(self.num_pos * mc.neg_pos_ratio,tf.int32)
        
        print ('self.num_neg:',self.num_neg)
        self.conf_loss = -tf.reduce_sum(gt_label*tf.log(conf_data_norm),axis=2)
        print ("conf_loss:",self.conf_loss)

        self.mine_hard_examples()
        self.cal_loc_loss(loc_data,prior_boxes,prior_variances,gt_boxes,input_mask)
        self.cal_conf_loss_v1()
        
        self.loss = tf.add(self.loc_loss,self.conf_loss)
        self._add_debug(self.conf_loss,"conf_loss")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
