# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:32:49 2020

@author: Majin
"""
import os
import cv2
import json
import random
import torch
from torch.backends import cudnn
import numpy as np
import logging as log
import yaml
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils_.utils import aspectaware_resize_padding, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

compound_coef = 2
force_input_size = None

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

threshold = 0.2
iou_threshold = 0.2
regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

#config_file = '/usr/local/ev_sdk/src/projects/helmet.yml'
#weights_path = '/usr/local/ev_sdk/model/efficientdet-d2.pth'
config_file = '/project/train/src_repo/motor/projects/motor.yml'
weights_path = '/project/train/models/final/efficientdet-d2.pth'

color_list = standard_to_bgr(STANDARD_COLORS)

def preprocess(image, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = [image]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

params = Params(config_file)
obj_list = params.obj_list

def init():
    """Initialize model

    Returns: model

    """ 
    # replace this part with your project's anchor config
    model = EfficientDetBackbone(num_classes=len(obj_list), compound_coef=compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
    
    model.load_state_dict(torch.load(weights_path))
    model.requires_grad_(False)
    model.eval()
    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    return model

def display(preds, imgs, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            if preds[i]['class_ids'][j]==0:
                continue
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(f'test_{i}.jpg', imgs[i])

def process_image(model, input_image):
    if input_image is None:
        log.error('Invalid input args')
        return None
    log.info(f'process_image, ({input_image.shape}')
#    ih, iw, _ = input_image.shape
    
    
    ori_imgs, framed_imgs, framed_metas = preprocess(input_image, max_size=input_size)
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
    
    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
    
    with torch.no_grad():
        _, regression, classification, anchors = model(x)

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
        out = invert_affine(framed_metas, out)
    
    return out, ori_imgs

if __name__ == '__main__':
    # Test API
    
    img_path = '/home/data/20'
    files_list = os.listdir(img_path)
    imgs_list = []
    for file in files_list:
        if file.endswith('.jpg'):
            imgs_list.append(file)
    print(len(imgs_list),'/',len(files_list))
    
    i = random.randint(0,len(imgs_list))
    img = cv2.imread(os.path.join(img_path,imgs_list[i]))
    
    model = init()
    results, imgs = process_image(model, img)
    log.info(results)
#    result, ori_imgs = process_image(model, img)
    print(len(results[0]['scores']))
    display(results, imgs, imshow=False, imwrite=True)