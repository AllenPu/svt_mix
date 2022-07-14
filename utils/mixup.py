import torch
import torch.nn as nn
import numpy as np
#from timm.data import Mixup
import torch.nn.functional as F

def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.
    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def data_mixup(x, lam):
    #
    # x in shape of : [B, C, T, H, W]
    # lam is the mixup ratio
    #
    # in our desired shape
    assert len(x.shape) == 5
    temporal_len = x.shape[2]
    #
    if lam is None:
        lam = np.random.beta(1., 1., size=[x.shape[0],1,1,1]).astype(np.float32)
        lam = torch.from_numpy(lam).cuda()
    new_lams = lam * 0.8 + 0.1
    x_flip = x.flip(0).clone()
    for i in range(x.shape[0]):
        # two video mixup in a temproal level       
        x_image = x[:, :, 0, :, :]
        yl, yh, xl, xh = rand_bbox(x_image.shape, new_lams[i].cpu())
        bbox_area = (yh - yl) * (xh - xl)
        new_lams[i] = torch.tensor(1. - bbox_area / float(x_image.shape[-2] * x_image.shape[-1])).cuda()
        # mix up for two videos frame by frame
        for j in range(temporal_len):
            x[i:i+1, :, j, yl:yh, xl:xh] = F.interpolate(x_flip[i:i+1, :, j, :, :], (yh - yl, xh - xl), mode="bilinear")
    # mixuped x, original ratio (first derievd from the beta distribution), new ratio after mixed up
    return x, lam, new_lams