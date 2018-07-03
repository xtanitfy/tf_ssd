import numpy as np
import math
import cv2

class Data_layer():
    def __init__(self, mc):
        self.mc = mc
        self.anno_box_filter_idx = []
   
    def run_transform(self,image,gt_boxes):
        pass
    
    def MeetEmitConstraint(self,clip_box,anno_boxes):
        anno_box_filter_idx = []
        for i,anno_box in enumerate(anno_boxes):
            xmin,ymin,xmax,ymax = anno_box
            x_center = (xmin + xmax) / 2.
            y_center = (ymin + ymax) / 2.
            
            clip_xmin,clip_ymin,clip_xmax,clip_ymax = clip_box
            if x_center > clip_xmin and x_center < clip_xmax \
                and y_center > clip_ymin and y_center < clip_ymax:
                anno_box_filter_idx.append(i)

        return anno_box_filter_idx

    def clip_boxes(self,boxes):
      for i in range(0,len(boxes)):
        boxes[i][0] = min(max(boxes[i][0],0.),1.)
        boxes[i][1] = min(max(boxes[i][1],0.),1.)
        boxes[i][2] = min(max(boxes[i][2],0.),1.)
        boxes[i][3] = min(max(boxes[i][3],0.),1.)
        
    def run_sampler(self,image,gt_boxes):
        mc  = self.mc
        batch_sampler = mc.batch_sampler
        sample_boxs = []
        
        image_bak = image
        gt_boxes_bak = gt_boxes 
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
        gt_boxes = np.array([gt_boxes[i] for i in anno_box_filter_idx])
        if len(gt_boxes) == 0:
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
        self.clip_boxes(gt_boxes)
        
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
   
    def exapnd(self,image,gt_boxes):
        mc = self.mc
        
        ori_height, ori_width, ori_channel = [int(v) for v in image.shape]
        gt_boxes[:,0::2] *= ori_width
        gt_boxes[:,1::2] *= ori_height
    
        b_expand = np.random.rand()
        if b_expand > mc.expand_param.prob:
            return image,gt_boxes
        
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
        
        return image_expand,gt_boxes
        
    def Process(self,image,gt_boxes):
        
        image,gt_boxes = self.exapnd(image,gt_boxes)
        self.draw_annno(image,gt_boxes,'out/expand.jpg')
        
        image,gt_boxes,anno_box_filter_idx = self.run_sampler(image,gt_boxes)
        self.draw_annno(image,gt_boxes,'out/run_sampler.jpg')
        
        image,gt_boxes = self.drift(image,gt_boxes)
        self.draw_annno(image,gt_boxes,'out/drift.jpg')
        
        return image,gt_boxes,self.anno_box_filter_idx
            
    def draw_annno(self,image,gt_boxes,filename):
        height, width, _ = [int(v) for v in image.shape]
        print (filename,' height:',height,' width:',width)
        
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
        
        gt_boxes = np.array([self.bbox_transform_inv(box) for box in gt_boxes])
        
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
        
        gt_boxes = np.array([self.bbox_transform(box) for box in gt_boxes])
        
        return im,gt_boxes
        
    def bbox_transform_inv(self,bbox):
        xmin, ymin, xmax, ymax = bbox
        out_box = [[]]*4

        width       = xmax - xmin + 1.0
        height      = ymax - ymin + 1.0
        out_box[0]  = xmin + 0.5*width 
        out_box[1]  = ymin + 0.5*height
        out_box[2]  = width
        out_box[3]  = height

        return out_box
        
    def bbox_transform(self,bbox):
        cx, cy, w, h = bbox
        out_box = [[]]*4
        out_box[0] = cx-w/2
        out_box[1] = cy-h/2
        out_box[2] = cx+w/2
        out_box[3] = cy+h/2

        return out_box 
        
        
        
        
        
        
        
        
        
        
        
        
    
     
     
     
     
       