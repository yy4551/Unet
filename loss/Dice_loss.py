"""
多分类Dice loss
使用最简单的策略：多个dice取平均
"""

import torch
import torch.nn as nn
class myDiceLoss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, pred, label):
            smooth = 1.

            iflat = pred.view(-1)
            tflat = label.view(-1)
            intersection = (iflat * tflat).sum()
            union = iflat.sum() + tflat.sum()


            return 1 - ((2. * intersection + smooth) /
                        (union + smooth))