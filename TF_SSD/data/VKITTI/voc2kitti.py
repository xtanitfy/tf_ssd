#!/usr/bin/python
# -*- coding: UTF-8 -*-

from xml.dom.minidom import parse
import xml.dom.minidom
import os
import shutil 

VOC_DIR='../VOC/VOCdevkit'
VKITTI_LABEL_DIR = 'training/label_2'
VKITTI_IMG_DIR = 'training/image_2'
VKITTI_IMGSET_DIR = 'ImageSets'

CLASS_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                       'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                       'sofa', 'train', 'tvmonitor')
                       
difficult_cnt=0                      
def parse_xml_file(xmlfilename,dst_dir,dst_filename):
    if xmlfilename.split('.')[-1] != 'xml':
        print (xmlfilename + ' is not xml file')
        return 
    
    DOMTree = xml.dom.minidom.parse(xmlfilename)
    annotation = DOMTree.documentElement

    filename = xmlfilename.split('/')[-1]
    #print kitti_file
    filename = annotation.getElementsByTagName('filename')[0]
    
    size = annotation.getElementsByTagName('size')[0]
    width = size.getElementsByTagName('width')[0]
    #print ('width:%d'%int(width.childNodes[0].data))
    height = size.getElementsByTagName('height')[0]
    #print ('height:%d'%int(height.childNodes[0].data))
    depth = size.getElementsByTagName('depth')[0]
    #print ('depth:%d'%int(depth.childNodes[0].data))

    objects = annotation.getElementsByTagName('object')
    
    objs = []
    
    global difficult_cnt
    for object in objects:
        #print ('===========')
        name = object.getElementsByTagName('name')[0]
        #print "name: %s" % name.childNodes[0].data 
        
        if name.childNodes[0].data not in CLASS_NAMES:
            continue
        str = ''
        str += name.childNodes[0].data + ' '
        
        difficult =  object.getElementsByTagName('difficult')[0]
        if int(difficult.childNodes[0].data) > 0:
            difficult_cnt += 1 
            continue 
        str += difficult.childNodes[0].data + ' ' 
        bndbox = object.getElementsByTagName('bndbox')[0]

        str += '-1 0.0 '
        xmin = bndbox.getElementsByTagName('xmin')[0]
        #print "xmin: %s" % xmin.childNodes[0].data
        #center_x = 

        str += xmin.childNodes[0].data + ' ' 

        ymin = bndbox.getElementsByTagName('ymin')[0]
        #print "ymin: %s" % ymin.childNodes[0].data
        str += ymin.childNodes[0].data + ' '

        xmax = bndbox.getElementsByTagName('xmax')[0]
        #print "xmax: %s" % xmax.childNodes[0].data
        str += xmax.childNodes[0].data + ' '

        ymax = bndbox.getElementsByTagName('ymax')[0]
        #print "ymax: %s" % ymax.childNodes[0].data
        str += ymax.childNodes[0].data + ' '

        str += '0.0 0.0 0.0 0.0 0.0 0.0 0.0\n'
        
        objs.append(str)
    
    if len(objs) == 0:
        return 0
        
    with open(os.path.join(dst_dir,dst_filename),"w+") as f:
        for item in objs:
            f.write(item)
        f.close()
    return len(objs)
    
def load_lable_img(txt_file,type):
    with open(txt_file,'r') as f:
        lines = f.readlines()
    
    if type == 'train':
        vkitti_txt = os.path.join(VKITTI_IMGSET_DIR,'train.txt')
    else:
        vkitti_txt = os.path.join(VKITTI_IMGSET_DIR,'val.txt')
    
    cnt = 0
    with open(vkitti_txt,'w') as f:
        for line in lines:
            items = line.strip().split(' ')
            img_file = os.path.join(VOC_DIR,items[0])
            shutil.copy(img_file,VKITTI_IMG_DIR)
            
            label_file = os.path.join(VOC_DIR,items[1])
            name = label_file.split('/')[-1].split('.')[0]
            kitti_label = name + '.txt'
            parse_xml_file(label_file,VKITTI_LABEL_DIR,kitti_label)
            
            f.write(name+'\n')
            
            print ('[{}] {}/{}'.format(type,cnt,len(lines)))
            cnt += 1
if __name__ == '__main__':
    load_lable_img('trainval.txt',type='train')
    load_lable_img('test.txt',type='val')
    
    
    
    
    
    
    