# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:23:14 2020

@author: Majin
"""

import numpy as np
#import glob
import xml.etree.ElementTree as ET
import cv2
import os
import argparse
# import yaml

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def get_xml_paths(root_dir):
    xml_paths = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.xml'):
                xml_paths.append(os.path.join(root, f))
    return xml_paths

def load_dataset(path, classes, show_img=False, category_id=None):
    dataset = []
    paths = get_xml_paths(path)
    # paths = [p.replace("\\", '/') for p in glob.glob("{}/*.xml".format(path))]
    gt_small, gt_mid, gt_large = 32*32, 96*96, float("inf")
    print("Get %d xmls" % len(paths))
    for xml_file in paths:
        tree = ET.parse(xml_file)
        # print(xml_file)
        img_file = xml_file.replace(".xml", ".jpg")
        image = cv2.imread(img_file)
        height, width, _ = image.shape

        # height = int(tree.findtext("./size/height"))
        # width = int(tree.findtext("./size/width"))
        # if H == height and W == width:
        #     print("Pass.")

        # To get absolute value, remove /width and height
        #for obj in tree.iter("object"):
        #    xmin = int(obj.findtext("bndbox/x0")) / width
        #    ymin = int(obj.findtext("bndbox/y0")) / height
        #    xmax = int(obj.findtext("bndbox/x1")) / width
        #    ymax = int(obj.findtext("bndbox/y1")) / height
        
        for obj in tree.iter("object"):
            if not obj.findtext('bndbox'):
                continue
            if obj.findtext('name') not in classes:
                continue
            xmin = int(obj.findtext("bndbox/xmin")) 
            ymin = int(obj.findtext("bndbox/ymin")) 
            xmax = int(obj.findtext("bndbox/xmax")) 
            ymax = int(obj.findtext("bndbox/ymax"))
            if category_id == 's':
                if 0<(xmax - xmin)*(ymax - ymin)<=gt_small:
                    dataset.append([xmax - xmin, ymax - ymin])
                    continue
            elif category_id == 'l':
                if gt_mid<(xmax - xmin)*(ymax - ymin):
                    dataset.append([xmax - xmin, ymax - ymin])
                    continue
            elif category_id == 'm':
                if gt_small<(xmax - xmin)*(ymax - ymin)<=gt_mid:
                    dataset.append([xmax - xmin, ymax - ymin])
                    continue
            else:
                # print("no category detected. Will use all possible bboxs")
                dataset.append([xmax - xmin, ymax - ymin])
            if show_img:
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,255,0), 1, cv2.LINE_AA)
            
        if show_img:
            cv2.imshow("loading image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    return np.array(dataset)

def compute_anchor_para(bboxs, anchor_base_scale = 4, anchor_stride = 8):
    """
    Compute anchor parameters, given all bboxes from kmean gathered
    Require anchor_base_scale, anchor_stride at first feature map, it depends on network configuration
    return anchor scale and anchor ratios
    default parameter should work for Resnet50 backbone
    """
    return_scale, return_ratio = [], []
    base_factor = anchor_base_scale * anchor_stride
    for height, width in bboxs:
        return_scale.append(height*1.0/base_factor)
        return_ratio.append((1,width*1.0/height))
    return return_scale, return_ratio

def get_args():
    parser = argparse.ArgumentParser('compute anchors')
    parser.add_argument('--data_path', type=str, default='/home/data/', help='the root folder of dataset')
    parser.add_argument('--yaml_file', type=str, default='../projects/motor.yml', help='config file name')
    args = parser.parse_args()
    return args

def changeYamlConfig(path, key, value):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
#        lines = []  
#        for line in f.readlines():
#            if line != '\n':
#                lines.append(line)
        f.close()
    with open(path, 'w', encoding='utf-8') as f:
        flag = 0
        for line in lines:
            if key in line and '#' not in line and line!='\n':
                leftstr=line.split(":")[0]
                newline = "{0}: {1}".format(leftstr, value)
                line = newline
                f.write('%s\n' % line)
                flag = 1
            else:
                f.write('%s' % line)
        f.close()
        return flag

if __name__=='__main__':
    opt = get_args()
    CLUSTERS = 3
    classes = ['motor']
    data = load_dataset(opt.data_path, classes, show_img=False)
    out = kmeans(data, k=CLUSTERS)
    
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    print("Boxes:\n {}".format(out))
    
    #ratios = np.around(1.0 / out[:, 0] * out[:, 1], decimals=2).tolist()
    #print("Ratios:\n {}".format(sorted(ratios)))
    scales, ratios = compute_anchor_para(out)
    
    ratios_ = []
    for ratio in ratios:
        ratio_ = [round(x,4) for x in ratio]
        ratios_.append(tuple(ratio_))
    scales_ = [round(x,4) for x in scales]
    print("computed scales: ", scales_)
    print("computed ratios: ", ratios_)

#    fy = "../projects/motor.yml"
    changeYamlConfig(opt.yaml_file, 'anchors_scales', '\''+str(scales_)+'\'')
    changeYamlConfig(opt.yaml_file, 'anchors_ratios', '\''+str(ratios_)+'\'')