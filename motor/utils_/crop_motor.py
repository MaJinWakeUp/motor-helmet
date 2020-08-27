# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:27:00 2020

@author: Majin
"""
import os
import cv2
import xml.etree.ElementTree as ET
from gen_xml import GEN_Annotations
import argparse
# crop images in dataset to get motor images, 
# for head and helmet detection
def crop_save(img, bbox, save_name):
    '''
    img: original image
    bbox: list[xmin, ymin, xmax, ymax]
    save_name: saved image name
    '''
    xmin, ymin, xmax, ymax = bbox
    cropped_img = img[ymin:ymax, xmin:xmax, :]
    cv2.imwrite(save_name, cropped_img)
    
def parser_xml(ann_path):
    tree = ET.parse(ann_path)
    root = tree.getroot()
    
    motors = []
    heads = []
    helmets = []
    
    for _object in root.findall('object'):
        if not _object.find('bndbox'):
            continue
        object_name = _object.find('name').text
        Xmin = int(float(_object.find('bndbox').find('xmin').text))
        Ymin = int(float(_object.find('bndbox').find('ymin').text))
        Xmax = int(float(_object.find('bndbox').find('xmax').text))
        Ymax = int(float(_object.find('bndbox').find('ymax').text))
        
        if object_name == 'motor':
            motors.append([Xmin, Ymin, Xmax, Ymax])
        elif object_name == 'head':
            heads.append([Xmin, Ymin, Xmax, Ymax])
        elif object_name == 'helmet':
            helmets.append([Xmin, Ymin, Xmax, Ymax])
        else:
            pass
    return motors, heads, helmets

def inside_motor(bbox1, bbox2, thres=0.5):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    if ymin1 > (ymax2+ymin2)/2:   # in bottom
        return False
    left = max(xmin1, xmin2)
    top = max(ymin1, ymin2)
    right = min(xmax1, xmax2)
    bottom = min(ymax1, ymax2)
    if left>=right or top>=bottom:  
        return False
    overlap_ratio = (right - left) * (bottom - top) / ((xmax1 - xmin1) * (ymax1 - ymin1))
    return thres <= overlap_ratio <= 1.0

def process_img(img_path, ann_path, save_folder):
    img = cv2.imread(img_path)
    motors, heads, helmets = parser_xml(ann_path)
    for i in range(len(motors)):
        motor_box = motors[i]
        file_name = os.path.splitext(os.path.basename(img_path))[0]+'_'+str(i)
        jpg_path = os.path.join(save_folder, file_name+'.jpg')
        xml_path = os.path.join(save_folder, file_name+'.xml')
        
        HAS_GT = False
        ann= GEN_Annotations(jpg_path)
        xmin_, ymin_, xmax_, ymax_ = motor_box
        ann.set_size(xmax_-xmin_, ymax_-ymin_,3)
        for j in range(len(heads)):
            head_box = heads[j]
            if inside_motor(head_box, motor_box):
                xmin,ymin,xmax,ymax = head_box
                xmin = max(0, xmin-xmin_)
                ymin = max(0, ymin-ymin_)
                xmax = min(xmax-xmin_, xmax_-xmin_)
                ymax = min(ymax-ymin_, ymax_-ymin_)
                ann.add_pic_attr("head",xmin,ymin,xmax,ymax)
                HAS_GT = True
        for k in range(len(helmets)):
            helmet_box = helmets[k]
            if inside_motor(helmet_box, motor_box):
                xmin,ymin,xmax,ymax = helmet_box
                xmin = max(0, xmin-xmin_)
                ymin = max(0, ymin-ymin_)
                xmax = min(xmax-xmin_, xmax_-xmin_)
                ymax = min(ymax-ymin_, ymax_-ymin_)
                ann.add_pic_attr("helmet",xmin,ymin,xmax,ymax)
                HAS_GT = True
        if HAS_GT:
            crop_save(img, motor_box, jpg_path)
            ann.savefile(xml_path)
        else:
            continue

def get_img_xml_paths(root_dir):
    img_paths = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.jpg'):
                img_paths.append(os.path.join(root, f))
    xml_paths = [os.path.splitext(x)[0]+'.xml' for x in img_paths]
    return img_paths, xml_paths

def get_args():
    parser = argparse.ArgumentParser('compute anchors')
    parser.add_argument('--data_path', type=str, default='/home/data/', help='the root folder of dataset')
    parser.add_argument('--save_path', type=str, default='/project/train/processed_data', help='save folder for processed images')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    opt = get_args()
    img_paths, xml_paths = get_img_xml_paths(opt.data_path)
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    for i in range(len(img_paths)):
        process_img(img_paths[i], xml_paths[i], opt.save_path)