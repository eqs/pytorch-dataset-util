# -*- coding: utf-8 -*-
import pandas as pd
from torchvision.datasets import VisionDataset


class VoTTCSVDataset(VisionDataset):
    def __init__(self, root, csv_path,
                 transforms=None, transform=None, target_transform=None):
        super(VoTTCSVDataset, self).__init__(root, transforms,
                                             transform, target_transform)

        self.data_table = pd.read_csv(csv_path)

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self):
        raise NotImplementedError()
