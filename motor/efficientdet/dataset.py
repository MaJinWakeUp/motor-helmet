import os
import torch
import numpy as np

from torch.utils.data import Dataset
# from pycocotools.coco import COCO
import xml.etree.ElementTree as ET
import cv2

def generate_sampler(trainset, portion=0.9):
    num_data = len(trainset)
    indices = torch.randperm(num_data).tolist()
    split_indice = int(num_data * portion)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split_indice])
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split_indice:])
    return train_sampler, val_sampler

class MotorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['background','motor']
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
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def load_image(self, image_index):
        img = cv2.imread(self.img_paths[image_index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

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

        return annotations
                        
def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
