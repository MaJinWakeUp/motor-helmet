import os
import os.path
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def generate_sampler(trainset, portion=0.9):
    num_data = len(trainset)
    indices = torch.randperm(num_data).tolist()
    split_indice = int(num_data * portion)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split_indice])
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split_indice:])
    return train_sampler, val_sampler

class HeadHelmetDataset(Dataset):
    def __init__(self, root_dir, preproc=None):
        self.root_dir = root_dir
        self.preproc = preproc
        self.classes = ['background', 'head', 'helmet']
        self.img_paths = self.get_img_paths(self.root_dir)
        
    def get_img_paths(self, root_dir):
        img_paths = []
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith('.jpg'):
                    img_paths.append(os.path.join(root, f))
        return img_paths
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        target = self.load_annotations(idx)
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        return torch.from_numpy(img), target
    
    def load_annotations(self, image_index):
        annotations = np.zeros((0, 5))
        xml_file = os.path.splitext(self.img_paths[image_index])[0]+'.xml'
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for _object in root.findall('object'):
            if not _object.find('bndbox'):
                continue
            object_name = _object.find('name').text
            if object_name not in self.classes:
                continue
            Xmin = int(float(_object.find('bndbox').find('xmin').text))
            Ymin = int(float(_object.find('bndbox').find('ymin').text))
            Xmax = int(float(_object.find('bndbox').find('xmax').text))
            Ymax = int(float(_object.find('bndbox').find('ymax').text))
            
            annotation = np.zeros((1, 5))
            annotation[0, :4] = [Xmin, Ymin, Xmax, Ymax]
            annotation[0, 4] = self.classes.index(object_name)
            annotations = np.append(annotations, annotation, axis=0)

        return np.array(annotations)
    
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
