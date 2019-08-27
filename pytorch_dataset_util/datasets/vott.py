# -*- coding: utf-8 -*-
import os

from PIL import Image
import numpy as np
import pandas as pd

import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor


class VoTTCSVDataset(VisionDataset):
    def __init__(self, root, csv_path,
                 transforms=None, transform=None, target_transform=None):
        super(VoTTCSVDataset, self).__init__(root, transforms,
                                             transform, target_transform)

        self.data_table = pd.read_csv(csv_path)

        self.filename_list = sorted(self.data_table['image'].unique())

        self._categories = \
            ['__background__'] + sorted(self.data_table.label.unique())
        self._cat2idx = {category: idx
                         for idx, category in enumerate(self._categories)}

    @property
    def categories(self):
        return self._categories

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):

        filepath = os.path.join(self.root, self.filename_list[idx])
        image = Image.open(filepath)
        target_table = self.data_table.query(
            'image == "{filename}"'.format(filename=self.filename_list[idx])
        )

        bboxes = np.array(target_table[['xmin', 'ymin', 'xmax', 'ymax']])
        labels = np.array([self._cat2idx[category]
                           for category in target_table['label']])

        target = {
            'boxes': torch.tensor(bboxes, dtype=float),
            'labels': torch.tensor(labels, dtype=int),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        image = ToTensor()(image)

        return image, target
