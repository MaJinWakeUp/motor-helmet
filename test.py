# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 15:58:59 2020

@author: Majin
"""
import os
import sys
import time
import random
import numpy as np
import cv2
sys.path.append('./head')
sys.path.append('./motor')
import json
from testm import init_motor, process_motor
from testh import init_head, process_head

classes_motor = ['background', 'motor']
classes_head = ['background', 'head', 'helmet']

images_path = '/home/data/20'

thres_motor = 0.5
thres_head = 0.8

def get_random_image(img_path):
    files_list = os.listdir(img_path)
    imgs_list = []
    for file in files_list:
        if file.endswith('.jpg'):
            imgs_list.append(file)
    print(len(imgs_list),'/',len(files_list))
    
    i = random.randint(0,len(imgs_list))
    img_raw = cv2.imread(os.path.join(img_path,imgs_list[i]),cv2.IMREAD_COLOR)
    return img_raw

def plot_one_box(image, box, score, cls):
    text = "{}:{:.4f}".format(cls, score)
    # print(text)
    x1, y1, x2, y2 = box.astype(np.int)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cx = x1
    cy = y1 + 5
    cv2.putText(image, text, (cx, cy),
                cv2.FONT_HERSHEY_DUPLEX, 0.2, (255, 255, 255))

def init():
    model = {}
    model['motor'] = init_motor()
    model['head'] = init_head()
    return model

def get_head(model, image, motors):
    ih, iw, _ = image.shape
    expand = 10
    num_motors = len(motors['rois'])
    motor_images = []
    refers = []
    for i in range(num_motors):
        x1, y1, x2, y2 = motors['rois'][i].astype(np.int)
        x1_ = max(0, x1 - expand)
        y1_ = max(0, y1 - expand)
        x2_ = min(iw, x2 + expand)
        y2_ = min(ih, y2 + expand)
        motor_images.append(image[y1_:y2_, x1_:x2_, :])
        refers.append(np.array([x1_, y1_, x1_, y1_]))
    results_head = process_head(model, motor_images)
    
    return results_head, refers

def process_image(model, image):
    time_0 = time.time()
    jsons = []
    results_motor = process_motor(model['motor'], image)
    # time_1 = time.time()
    # print(f'using {time_1-time_0} seconds for motor detection!')
    if len(results_motor['rois'])==0:
        # cv2.imwrite('test.jpg', image)
        return json.dumps({"objects": jsons})
    else:
        results_head, refers = get_head(model['head'], image, results_motor)
        # time_2 = time.time()
        # print(f'using {time_2-time_1} seconds for head detection!')
        for i in range(len(results_motor['rois'])):
            box = results_motor['rois'][i]
            name = classes_motor[results_motor['class_ids'][i]]
            score = float(results_motor['scores'][i])
            if name == 'background' or score < thres_motor:
                continue
            obj = {
                    'name': name,
                    'xmin': int(box[0]),
                    'ymin': int(box[1]),
                    'xmax': int(box[2]),
                    'ymax': int(box[3]),
                    'confidence': score
                    }
            jsons.append(obj)
            # plot_one_box(image, box, score, name)
            
            refer = refers[i]
            for j in range(len(results_head['rois'][i])):
                box = results_head['rois'][i][j] + refer
                name = classes_head[results_head['class_ids'][i][j]]
                score = float(results_head['scores'][i][j])
                if name == 'background' or score < thres_head:
                    continue
                obj = {
                    'name': name,
                    'xmin': int(box[0]),
                    'ymin': int(box[1]),
                    'xmax': int(box[2]),
                    'ymax': int(box[3]),
                    'confidence': score
                    }
                jsons.append(obj)
                # plot_one_box(image, box, score, name)
        # time_3 = time.time()
        # print(f'using {time_3-time_2} seconds for data processing!')
        # cv2.imwrite('test.jpg', image)
        return json.dumps({"objects": jsons})
    
if __name__=='__main__':
    image = get_random_image(images_path)
    model = init()
    json_data = process_image(model, image)
    print(json_data)
    