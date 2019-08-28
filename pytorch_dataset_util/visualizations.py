# -*- coding: utf-8 -*-
import numpy as np
from chainercv.visualizations import vis_bbox as chainer_vis_bbox


def vis_bbox(img, bbox, label=None, score=None, label_names=None, ax=None):
    """A wrapper of chainer function for visualizing bbox inside image.

    Args:
        img (~torch.tensor):
            an image which shape :math:`(3, H, W)` and format RGB [0.0, 1.0]
        bbox (~torch.tensor):
            bounding boxes we want to show.
            Its shape is :math:`(N, 4)` and format is :math:`(x_\\mathrm{min}, y_\\mathrm{min}, \
            x_\\mathrm{max}, y_\\mathrm{max})`
        label (~torch.tensor):
            labels of each bbox
        score (~torch.tensor):
            scores of each bbox
        label_names (iterable of strings):
            Name of labels ordered according to label ids.
            If this is `None`, labels will be skipped.
    """

    return chainer_vis_bbox(np.uint8(img * 255), bbox[:, [1, 0, 3, 2]],
                            label=label, score=score,
                            label_names=label_names, ax=ax)
