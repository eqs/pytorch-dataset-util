# -*- coding: utf-8 -*-
import os
import glob

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor

from .utils import _download_file, _extract_zip

__all__ = ['UECFOOD100']

class UECFOOD100Dataset(VisionDataset):
    
    url = 'http://foodcam.mobi/dataset100.zip'

    def __init__(self, root, 
                 transform=None, 
                 target_transform=None, 
                 transforms=None, download=False):
        super(UECFOOD100Dataset, self).__init__(root, transforms,
                                                transform, target_transform)
        
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. ' +
                               'You can use download=True to download it.')
            
        self._load_images_and_bboxes()
    
    @property
    def data_directory(self):
        return os.path.join(self.root, 'UECFOOD100')
    
    @property
    def categories(self):
        return self._categories

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        get_image_id = lambda x: int(os.path.splitext(os.path.split(x)[1])[0])
        
        image_id = get_image_id(self.images[idx])
        bbox_in_image = self.bboxes.query(f'img == {image_id}')
        
        image = np.array(Image.open(self.images[idx]).convert('RGB'))
        
        labels = [list(b[['category']]) for row, b in bbox_in_image.iterrows()]
        bbs = [list(b[['x1', 'y1', 'x2', 'y2']]) for row, b in bbox_in_image.iterrows()]
        
        target = {
            'labels': torch.tensor(labels, dtype=torch.int64).flatten(),
            'boxes': torch.tensor(bbs, dtype=torch.float32)
        }
        
        image = ToTensor()(image)
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target

    def download(self):

        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
            print(f'{self.data_directory} is created.')
        
        _download_file(self.url, self.root)
        _extract_zip(os.path.join(self.root, 'dataset100.zip'), self.root)
    
    def _check_exists(self):
        return os.path.exists(self.data_directory)
    
    def _load_images_and_bboxes(self):
        
        _categories = pd.read_csv(os.path.join(self.data_directory, 'category.txt'), 
                                  delimiter='\t')
        self._categories = ['__background__'] + _categories['name'].tolist()
        
        images_path = glob.glob(os.path.join(self.data_directory, '*/*.jpg'))
        bboxes_path = glob.glob(os.path.join(self.data_directory, '*/bb_info.txt'))
        
        # Make images unique
        images_name = np.array(list(map(lambda x: os.path.split(x)[1], images_path)))
        images_name_unique, unique_idx = np.unique(images_name, return_index=True)

        images_path_unique = np.array(images_path)[unique_idx].tolist()
        
        get_image_id = lambda x: int(os.path.splitext(os.path.split(x)[1])[0])
        images_path_unique = sorted(images_path_unique, key=get_image_id)
        
        # Create BBox table
        get_bbox_category = lambda x: int(os.path.split(os.path.split(x)[0])[1])
        bboxes_path = sorted(bboxes_path, key=get_bbox_category)
        bboxes_categories = list(map(get_bbox_category, bboxes_path))
        
        bboxes_dfs = [pd.read_csv(path, delimiter=' ') for path in bboxes_path]
        
        for df, cat in zip(bboxes_dfs, bboxes_categories):
            df.insert(1, 'category', cat)
            
        self.bboxes = pd.concat(bboxes_dfs, axis=0)
        self.images = images_path_unique


