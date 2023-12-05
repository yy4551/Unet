"""
多分类Dice loss
使用最简单的策略：多个dice取平均
"""
from dataset.dataset import output_tensor_label,output_tensor_pred
import torch
import torch.nn as nn
from Log import logger
from torch.utils.data import Dataset as dataset

import torch
from torch.utils.data import Dataset


class myDiceLoss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, pred, label):
            smooth = 1.

            #make iflat.max()=1,so that it's a labelmap and the DiceLoss won't be greater than 1
            pred = torch.clamp(pred, max=1)

            # Flatten both tensor to 1 dim array
            iflat = pred.view(-1)
            logger.info("iflatmax:{},iflatmin:{}".format(iflat.max(),iflat.min()))
            tflat = label.view(-1)
            logger.info("tflatmax:{},tflatmin:{}".format(tflat.max(),tflat.min()))

            intersection = (iflat * tflat).sum()
            logger.info("intersection:{}".format(intersection))
            union = iflat.sum() + tflat.sum()
            logger.info("union:{}".format(union))

            # if iflat.max() is above a hundard large,then I could greater than U easily,
            # then DiceLoss will Greater than 1 and loss will be negative
            DiceLoss = ((2. * intersection + smooth) /
                        (union + smooth))
            logger.info("DiceLoss:{}".format(DiceLoss))

            return 1 - DiceLoss

if __name__ == '__main__':
    pred = torch.randn(3, 3)
    label = torch.randn(3, 3)
    loss = myDiceLoss()
    print(loss(pred, label))