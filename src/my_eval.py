# -*- coding: utf-8 -*-
import os
import numpy as np
from utils.util import bbox_transform_inv, iou,iou_v1

class Eval():
    def __init__(self, class_name,gt_dir,dt_dir,score_threshold=0.4,iou_threshold=0.45,backgroud_id=0):
        self.class_name = class_name
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        self.all_correct_num = 0
        self.all_label_num = 0
        self.all_loss_num = 0
        self.mean_ap = 0
        self.recal = 0
          
        self.aps = np.zeros(len(self.class_name)) 
        self.correct_num = np.zeros(len(self.class_name)) 
        self.det_num = np.zeros(len(self.class_name)) 
        self.label_num = np.zeros(len(self.class_name)) 
        self.loss_num = np.zeros(len(self.class_name)) 

        self.gt_box_average_size = 0
        self.min_box_size = 0.
        self.max_box_size = 40*80

        self.small_num = 0
        self.dt_small_num = 0
          
        self.backgroud_id = backgroud_id
        self.gt_dir = gt_dir
        self.dt_dir = dt_dir

    def parse_lines(self,lines):
        names = []
        boxes = []
        scores = []
        for line in lines:
            items = line.strip().split(' ')
            names.append(items[0])
            boxes.append([float(items[4]),float(items[5]),float(items[6]),float(items[7])])
            scores.append(float(items[-1]))
        return names,boxes,scores
          
    def cal_detection_res(self):
        all_det_num = 0
        for i in range(0,len(self.class_name)):
            # backgroud
            if i == self.backgroud_id:
              continue
                
            print ('====================')
            print ('{} correct_num:{}'.format(self.class_name[i],self.correct_num[i]))
            print ('{} det_num:{}'.format(self.class_name[i],self.det_num[i]))
            print ('{} loss_num:{}'.format(self.class_name[i],self.loss_num[i]))
            print ('{} label_num:{}'.format(self.class_name[i],self.label_num[i]))
             
            assert self.det_num[i] != 0
            
            self.aps[i] = self.correct_num[i] / self.det_num[i]
            self.mean_ap += self.aps[i]
            self.all_correct_num += self.correct_num[i]
            self.all_label_num += self.label_num[i]
            self.all_loss_num += self.loss_num[i]
            all_det_num += self.det_num[i]
            print ('{} ap:{}'.format(self.class_name[i],self.aps[i]))
            
        self.mean_ap /= (len(self.class_name) - 1)
          
        print ('all_label_num:',self.all_label_num)
        print ('all_loss_num:',self.all_loss_num)
        print ('all_correct_num:',self.all_correct_num)
        print ('all_det_num:',all_det_num)

        print ('all_small_num:',self.small_num)
        print ('all_dt_small_num:',self.dt_small_num)
          
        self.small_recall = self.dt_small_num / float(self.small_num)
        self.recall = (self.all_label_num - self.all_loss_num)/ float(self.all_label_num)
        print ('mean_ap:{} recall:{} small_recall:{}'.format(self.mean_ap,self.recall,self.small_recall))
          
        return self.mean_ap,self.recall

    def analyze_det_V1(self,dt_names,dt_boxes,dt_scores,gt_names,gt_boxes):
        dt_names = np.array(dt_names)
        dt_boxes = np.array(dt_boxes)
        dt_scores = np.array(dt_scores)

        gt_names = np.array(gt_names)
        gt_boxes = np.array(gt_boxes)

        for c in range(0,len(self.class_name)):
            if c == self.backgroud_id:
              continue
            
            if len(dt_names) == 0 and len(gt_names) == 0:
              continue
                
            dt_cls_order = np.where(dt_names == self.class_name[c])
            gt_cls_order = np.where(gt_names == self.class_name[c])
            
            cls_dt_name = dt_names[dt_cls_order]
            cls_dt_boxes = dt_boxes[dt_cls_order]
            cls_dt_scores = dt_scores[dt_cls_order]
            
            score_filter = np.where(cls_dt_scores >= self.score_threshold)
            cls_dt_name = cls_dt_name[score_filter]
            cls_dt_boxes = cls_dt_boxes[score_filter]
            cls_dt_scores = cls_dt_scores[score_filter]

            cls_gt_name = gt_names[gt_cls_order]
            cls_gt_boxes = gt_boxes[gt_cls_order]

            self.det_num[c] += len(cls_dt_name)
            self.label_num[c] +=  len(cls_gt_name)
            
            #首先计算漏检的
            find = np.zeros((len(cls_gt_name)))
            for i in range(0,len(cls_gt_name)):
                
              for j in range(0,len(cls_dt_name)):
                    iou_val = iou_v1(cls_gt_boxes[i],cls_dt_boxes[j])
                    if iou_val >= self.iou_threshold:
                        find[i] = 1
                        boxsize = self.box_size(cls_gt_boxes[i])
                        if boxsize >= self.min_box_size and boxsize < self.max_box_size:
                             self.dt_small_num += 1
                        
            self.loss_num[c] += len(find[np.where(find == 0)])

            
            #计算准确率
            correct = np.zeros((len(cls_dt_name)))
            for i in range(0,len(cls_dt_name)):
              for j in range(0,len(cls_gt_name)):
                    iou_val = iou_v1(cls_dt_boxes[i],cls_gt_boxes[j])
                    if iou_val >= self.iou_threshold:
                        correct[i] = 1
                        break
                        
            self.correct_num[c] += len(correct[np.where(correct == 1)])
      
    def analyze_det(self,dt_names,dt_boxes,dt_scores,gt_names,gt_boxes):
        dt_names = np.array(dt_names)
        dt_boxes = np.array(dt_boxes)
        dt_scores = np.array(dt_scores)

        gt_names = np.array(gt_names)
        gt_boxes = np.array(gt_boxes)
          
        for c in range(0,len(self.class_name)):
            if c == self.backgroud_id:
              continue

            if len(dt_names) == 0 and len(gt_names) == 0:
              continue
            
            dt_cls_order = np.where(dt_names == self.class_name[c])
            gt_cls_order = np.where(gt_names == self.class_name[c])
            
            cls_dt_name = dt_names[dt_cls_order]
            cls_dt_boxes = dt_boxes[dt_cls_order]
            cls_dt_scores = dt_scores[dt_cls_order]
            
            score_filter = np.where(cls_dt_scores >= self.score_threshold)
            cls_dt_name = cls_dt_name[score_filter]
            cls_dt_boxes = cls_dt_boxes[score_filter]
            cls_dt_scores = cls_dt_scores[score_filter]

            cls_gt_name = gt_names[gt_cls_order]
            cls_gt_boxes = gt_boxes[gt_cls_order]
            
            self.det_num[c] += len(cls_dt_name)
            self.label_num[c] +=  len(cls_gt_name)
            
            if len(cls_dt_name) == 0 and len(cls_gt_name) != 0:
              self.loss_num[c] += len(cls_gt_name)    
              continue 
                
            if len(cls_gt_name) == 0:
              continue
                
            no_loss = np.zeros((len(cls_gt_name)))
            for i in range(0,len(cls_dt_name)):       
              for j in range(0,len(cls_gt_name)):
                      #iou_val = iou(bbox_transform_inv(cls_dt_boxes[i]),
                      #             bbox_transform_inv(cls_gt_boxes[j]))
                    iou_val = iou_v1(cls_dt_boxes[i],cls_gt_boxes[j])
                    if iou_val >= self.iou_threshold:
                        self.correct_num[c] += 1
                        no_loss[j] = 1
                        break
                        
            self.loss_num[c] += len(no_loss[np.where(no_loss == 0)]) 

            
            
    def box_size(self,box):
        return (box[2] - box[0]) * (box[3] - box[1])
          
    def evaluate_det(self):
        det_files = os.listdir(self.dt_dir)
        all_num = len(det_files)
        cnt = 0

        boxes_cnt = 0
          
        for filename in det_files:
            
            gt_file = self.gt_dir + '/' + filename
            if cnt % 100 == 0:
              print ('{}/{}'.format(cnt,all_num))
            #print ('==================gt_file:',gt_file)
            assert os.path.isfile(gt_file),'Not exist {}'.format(gt_file)
            
            dt_file = self.dt_dir + '/' + filename

            #print ('==================gt_file:',gt_file)
            #print ('==================dt_file:',dt_file)
            with open(gt_file, 'r') as f:
              gt_lines = f.readlines()    
                
            with open(dt_file, 'r') as f:
              dt_lines = f.readlines()
            
            dt_names,dt_boxes,dt_scores = self.parse_lines(dt_lines)
            gt_names,gt_boxes,_ = self.parse_lines(gt_lines)
            #print ('dt_names:',dt_names)
            #print ('dt_boxes:',dt_boxes)
            #print ('dt_scores:',dt_scores)

            #raw_input('pause')
            
            #print ('gt_names:',gt_names)
            #print ('gt_boxes:',gt_boxes)
            
            
            self.analyze_det_V1(dt_names,dt_boxes,dt_scores,gt_names,gt_boxes)
            
            boxes_size = np.array([self.box_size(box) for box in gt_boxes])
            all_cnt = boxes_cnt+len((boxes_size))
            self.gt_box_average_size = self.gt_box_average_size * (boxes_cnt/float(all_cnt)) + np.sum(boxes_size)/float(all_cnt) 

            order = np.where(boxes_size < self.max_box_size)
            boxes_size = boxes_size[order]

            order = np.where(boxes_size >= self.min_box_size)
            boxes_size = boxes_size[order]
            
            self.small_num += len(boxes_size)
            #print ('boxes_size:',boxes_size)
            #print ('gt_box_average_size:',self.gt_box_average_size)
            #raw_input('pause')
            boxes_cnt += len(boxes_size)
            cnt += 1

        print ('self.gt_box_average_size:',self.gt_box_average_size)
        print ('self.small_num:',self.small_num)
          
        return self.cal_detection_res()         
        