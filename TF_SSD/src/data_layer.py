import numpy as np
import math
import cv2
from dataset.imdb  import imdb
from utils.util import bbox_transform,bbox_transform_inv
from dataset import kitti


class Data_layer(kitti):
    def __init__(self, image_set, data_path, mc):
        kitti.__init__(self, image_set, data_path,mc)
        self.mc = mc
        self.anno_box_filter_idx = []
        
    def run_transform(self,image,gt_boxes):
        pass
    
    def MeetEmitConstraint(self,clip_box,anno_boxes):
        anno_box_filter_idx = []
        for i,anno_box in enumerate(anno_boxes):
            xmin,ymin,xmax,ymax = anno_box
            xmin = min(max(xmin,0.),1.)
            ymin = min(max(ymin,0.),1.)
            xmax = min(max(xmax,0.),1.)
            ymax = min(max(ymax,0.),1.)
            
            x_center = (xmin + xmax) / 2.
            y_center = (ymin + ymax) / 2.
            
            clip_xmin,clip_ymin,clip_xmax,clip_ymax = clip_box
            if x_center > clip_xmin and x_center < clip_xmax \
                and y_center > clip_ymin and y_center < clip_ymax:
                anno_box_filter_idx.append(i)

        return anno_box_filter_idx

    def clip_boxes(self,boxes):
      box_new = []
      for box in boxes:
        xmin,ymin,xmax,ymax = box
        xmin = min(max(xmin,0.),1.)
        ymin = min(max(ymin,0.),1.)
        xmax = min(max(xmax,0.),1.)
        ymax = min(max(ymax,0.),1.)
        box_new.append([xmin,ymin,xmax,ymax])
      return np.array(box_new)

        
    def run_sampler(self,image,gt_boxes):
        mc  = self.mc
        batch_sampler = mc.batch_sampler
        sample_boxs = []
        
        image_bak = image.copy()
        gt_boxes_bak = gt_boxes.copy() 
        for i in range(1,len(batch_sampler)):
            sampler = batch_sampler[i]
            found = 0
            max_sample = sampler['max_sample']
            for cnt in range(0,sampler['max_trials']):
                if found >=  max_sample:
                    break
                scale = self.random_float(sampler['min_scale'],sampler['max_scale'])
                aspect_ratio = self.random_float(sampler['min_aspect_ratio'],sampler['max_aspect_ratio'])
                aspect_ratio = max(aspect_ratio,math.pow(scale,2))
                aspect_ratio = min(aspect_ratio,1./math.pow(scale,2))
                bbox_width = scale * math.sqrt(aspect_ratio)
                bbox_height = scale / math.sqrt(aspect_ratio)
                
                w_off = self.random_float(0.,1-bbox_width)
                h_off = self.random_float(0.,1-bbox_height)
                sample_box = [w_off,h_off,w_off+bbox_width,h_off+bbox_height]
                
                min_jaccard_overlap = sampler['min_jaccard_overlap']
                max_jaccard_overlap = sampler['max_jaccard_overlap']
                
                for gt_box in gt_boxes:
                    iou_val = self.iou(sample_box,gt_box)
                    if iou_val >= min_jaccard_overlap or iou_val <= max_jaccard_overlap: 
                        found += 1
                        sample_boxs.append(sample_box)
        
        rng_id = np.random.randint(1000000) % len(sample_boxs) 
        sample_box = sample_boxs[rng_id]

        
        anno_box_filter_idx = self.MeetEmitConstraint(sample_box,gt_boxes)

        #print ('len(self.anno_box_filter_idx):',len(anno_box_filter_idx))
        #print ('len(gt_boxes):',len(gt_boxes))
        
        gt_boxes = np.array([gt_boxes[i] for i in anno_box_filter_idx])
        if len(gt_boxes) == 0:
            anno_box_filter_idx = [i for i in range(len(gt_boxes_bak))]
            return image_bak,gt_boxes_bak,anno_box_filter_idx
        
        gt_boxes[:,0::2] -= sample_box[0]
        gt_boxes[:,1::2] -= sample_box[1]

        img_height, img_width, _ = [int(v) for v in image.shape]
        gt_boxes[:,0::2] *= img_width
        gt_boxes[:,1::2] *= img_height
        
        xmin = int(sample_box[0] * img_width)
        ymin = int(sample_box[1] * img_height)
        xmax = int(sample_box[2] * img_width)
        ymax = int(sample_box[3] * img_height)
        
        new_image = image[ymin:ymax,xmin:xmax,:]
        
        new_img_height, new_img_width, _ = [int(v) for v in new_image.shape]
        gt_boxes[:,0::2] /= new_img_width
        gt_boxes[:,1::2] /= new_img_height

        #gt_boxes[:,:] = min(max(gt_boxes[:,:],0.),1.)
        gt_boxes = self.clip_boxes(gt_boxes)
        
        return new_image,gt_boxes,anno_box_filter_idx
                    
                    
    def iou(self,bbox1, bbox2):
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
    
    def random_float(self,min,max):
        if min >= max:
            return min
        return np.random.randint(min*1000000,max*1000000)/1000000.

    def expand(self,image,gt_boxes):
        mc = self.mc
        b_expand = np.random.rand()
        if b_expand > mc.expand_param.prob:
            return image,gt_boxes
            
        ori_height, ori_width, ori_channel = [int(v) for v in image.shape]
        
        gt_boxes[:,0::2] *= ori_width
        gt_boxes[:,1::2] *= ori_height
        
        expand_ratio = np.random.randint(mc.expand_param.min_expand_ratio,mc.expand_param.max_expand_ratio)

        height = int(ori_height * expand_ratio)
        width  = int(ori_width * expand_ratio)
        
        if height != ori_height:
            h_off = self.random_float(0,height - ori_height)
            w_off = self.random_float(0,width - ori_width)
            h_off = int(np.floor(h_off))
            w_off = int(np.floor(w_off))
        else:
            h_off = 0
            w_off = 0
           
        image_expand = np.zeros((height, width, ori_channel))
        image_expand[h_off:h_off+ori_height,w_off:w_off+ori_width,:] = image[:,:,:]
        
        gt_boxes[:,0::2] += w_off
        gt_boxes[:,1::2] += h_off
        
        gt_boxes[:,0::2] /= width
        gt_boxes[:,1::2] /= height
        
        #print ('exapnd gt_boxes:',gt_boxes)
        return image_expand,gt_boxes
        
    def Preprocess(self,image,gt_boxes):
        
        image,gt_boxes = self.expand(image,gt_boxes)
        #self.draw_annno(image,gt_boxes,'out/expand.jpg')
        
        image,gt_boxes,anno_box_filter_idx = self.run_sampler(image,gt_boxes)
        #self.draw_annno(image,gt_boxes,'out/run_sampler.jpg')
        
        #image,gt_boxes = self.drift(image,gt_boxes)
        #self.draw_annno(image,gt_boxes,'out/drift.jpg')

        #anno_box_filter_idx = [i for i in range(0,len(gt_boxes))]
        return image,gt_boxes,anno_box_filter_idx

    def parse_gt_data(self,gt_data):
        mc = self.mc
        gt_boxes = []
        gt_labels = []
        for i in range(0,mc.BATCH_SIZE):
            gt_boxes.append([])
            gt_labels.append([])

        for i in range(0,len(gt_data)):
            batch_id = int(gt_data[i][0])
            assert batch_id < mc.BATCH_SIZE
            gt_boxes[batch_id].append([gt_data[i][3],gt_data[i][4],gt_data[i][5],gt_data[i][6]])
            assert gt_data[i][1] < mc.CLASSES

            gt_labels[batch_id].append(gt_data[i][1])

        #print ('[parse_gt_data]gt_boxes:',gt_boxes)
        #print ('[parse_gt_data]gt_labels:',gt_labels)
        
        return gt_boxes,gt_labels
        
    def Get_feed_data(self):
        mc = self.mc
        batch_gt_boxes,batch_gt_labels,batch_image = self.read_batch_gt_data(shuffle=True)

        batch_gt_boxes = np.array(batch_gt_boxes)
        batch_gt_labels = np.array(batch_gt_labels)
        batch_image = np.array(batch_image)
        
        input_images = []

        gt_data = []
        for i in range(0,len(batch_gt_boxes)):
          im = batch_image[i]
          im -= mc.BGR_MEANS
          gt_bbox = np.array(batch_gt_boxes[i])
          gt_label = np.array(batch_gt_labels[i])

          im,gt_bbox,anno_box_filter_idx = self.Preprocess(im,gt_bbox)  
          assert len(anno_box_filter_idx) == len(gt_bbox)
          
          lables = []
          for idx in anno_box_filter_idx:
            lables.append(gt_label[idx])
          
          #lables = [[gt_label[idx]] for idx in anno_box_filter_idx]
          gt_label = np.array(lables)
          orig_h, orig_w, _ = [float(v) for v in im.shape]

          #mirror
          gt_bbox[:,0::2] *= orig_w
          gt_bbox[:,1::2] *= orig_h
          gt_bbox_center = np.array([bbox_transform_inv(box) for box in gt_bbox])
          if np.random.randint(2) > 0.5:
            im = im[:, ::-1, :]
            gt_bbox_center[:, 0] = orig_w - 1 - gt_bbox_center[:, 0]
          gt_bbox = np.array([bbox_transform(box) for box in gt_bbox_center])
          gt_bbox[:,0::2] /= orig_w
          gt_bbox[:,1::2] /= orig_h
          
          im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
          input_images.append(im)
          
          # scale image
          #image_anno = im + mc.BGR_MEANS
          #self.draw_annno(image_anno,gt_bbox,'test_' + str(i) + '.jpg')
          #gt_data.append([i,])
          num = len(gt_bbox)
          
          for j in range(0,num):
            gt_data.append([i,gt_label[j],0,gt_bbox[j][0],gt_bbox[j][1],gt_bbox[j][2],gt_bbox[j][3]])
            
          #batch_ids = np.ones((num ,1))*i
          #instance_ids = np.ones((num ,1))
          #gt_data.append(np.concatenate([batch_ids,gt_label,instance_ids,gt_bbox],axis=1))

        gt_boxes,gt_labels = self.parse_gt_data(gt_data)
        
        all_match_indices,all_match_overlaps = self._math_bbox(mc.ANCHOR_BOX,gt_boxes)
        gt_boxes_dense,gt_labels_dense,input_mask = self._sparse_to_dense(gt_boxes,gt_labels,all_match_indices)
        
        return input_images,gt_boxes_dense,gt_labels_dense,input_mask,all_match_overlaps
       
    def draw_annno(self,image,gt_boxes,filename):
        height, width, _ = [int(v) for v in image.shape]
        #print (filename,' height:',height,' width:',width)
        
        gt_boxes[:,0::2] *= width
        gt_boxes[:,1::2] *= height
        for box in gt_boxes:
            box_1 = [int(box[i]) for i in range(0,len(box))]
            cv2.rectangle(image, (box_1[0], box_1[1]), (box_1[2], box_1[3]), (255,0,0), 2)
        
        gt_boxes[:,0::2] /= width
        gt_boxes[:,1::2] /= height
        cv2.imwrite(filename,image)
    
    def drift(self,image,gt_boxes):
        mc = self.mc
        
        drift_prob = np.random.rand()
        if drift_prob > mc.DRIFT_PROB:
            return image,gt_boxes 
        
        gt_boxes = np.array([bbox_transform_inv(box) for box in gt_boxes])
        
        # Ensures that gt boundibg box is not cutted out of the image
        max_drift_x = min(gt_boxes[:, 0] - gt_boxes[:, 2]/2.0+1)
        max_drift_y = min(gt_boxes[:, 1] - gt_boxes[:, 3]/2.0+1)
        assert max_drift_x >= 0 and max_drift_y >= 0, 'bbox out of image'

        dy = np.random.randint(-mc.DRIFT_Y, min(mc.DRIFT_Y+1, max_drift_y))
        dx = np.random.randint(-mc.DRIFT_X, min(mc.DRIFT_X+1, max_drift_x))

        # shift bbox
        gt_boxes[:, 0] = gt_boxes[:, 0] - dx
        gt_boxes[:, 1] = gt_boxes[:, 1] - dy
        
        orig_h, orig_w, _ = [int(v) for v in image.shape]
        # distort image
        orig_h -= dy
        orig_w -= dx
        orig_x, dist_x = max(dx, 0), max(-dx, 0)
        orig_y, dist_y = max(dy, 0), max(-dy, 0)

        distorted_im = np.zeros((int(orig_h), int(orig_w), 3)).astype(np.float32)
        distorted_im[dist_y:, dist_x:, :] = image[orig_y:, orig_x:, :]
        im = distorted_im
        
        gt_boxes = np.array([bbox_transform(box) for box in gt_boxes])
        
        return im,gt_boxes
    
    def iou(self,bbox1, bbox2):
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
        
    def _math_bbox(self,prior_boxes,gt_boxes):
        mc = self.mc
        NUM_PRIORBOX = mc.ANCHORS_NUM
        
        all_match_indices = np.zeros((mc.BATCH_SIZE,NUM_PRIORBOX)) - 1
        all_match_overlaps = np.zeros((mc.BATCH_SIZE,NUM_PRIORBOX))
        
        for n in range(0,mc.BATCH_SIZE):
        
            overlaps = np.zeros((NUM_PRIORBOX,len(gt_boxes[n])))
            for i in range(0,NUM_PRIORBOX):
                for j in range(0,len(gt_boxes[n])):
                    overlap = self.iou(prior_boxes[i],gt_boxes[n][j])
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
        
    def _sparse_to_dense(self,gt_boxes,gt_labels,all_match_indices):
        mc = self.mc
        NUM_PRIORBOX = mc.ANCHORS_NUM
        gt_boxes_dense = np.zeros([mc.BATCH_SIZE, NUM_PRIORBOX,4])
        gt_labels_dense = np.zeros([mc.BATCH_SIZE, NUM_PRIORBOX,mc.CLASSES])
        input_mask = np.zeros([mc.BATCH_SIZE, NUM_PRIORBOX])

        for n in range(0,mc.BATCH_SIZE):
            for i in range(0,NUM_PRIORBOX):
                one_label = np.zeros(mc.CLASSES)
                one_box = np.zeros(4)
                if all_match_indices[n][i] == -1:
                    one_label[mc.background_label_id] = 1 
                else:
                    input_mask[n][i] = 1
                    gt_idx = int(all_match_indices[n][i])
                    label_idx = int(gt_labels[n][gt_idx])
                    assert label_idx < mc.CLASSES
                    one_label[label_idx] = 1
                    one_box = gt_boxes[n][gt_idx]
                gt_labels_dense[n][i] = one_label
                gt_boxes_dense[n][i] = one_box
        
        return gt_boxes_dense,gt_labels_dense,input_mask 
        