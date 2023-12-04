"""
随机取样方式下的数据集
"""

import os
import random

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from torch.utils.data import DataLoader

on_server = False
# size = 48

# notice:inherited from "dataset"
class Dataset(dataset):

    # acquire the names of every training and testing image,concatenate each with its directory
    def __init__(self, ct_dir, seg_dir):

        self.ct_list = os.listdir(ct_dir)
        self.seg_list = os.listdir(seg_dir)
        # self.seg_list = list(map(lambda x: x.replace('avg', 'avg_seg'), self.ct_list))

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))


    def __getitem__(self, index):
        """
        :param index:
        :return: torch.Size([B, 1, 48, 256, 256]) torch.Size([B, 48, 256, 256])
        """

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        # 将CT和金标准读入到内存中
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        # # 在slice平面内随机选取48张slice
        # # here x (ct_array.shape[0]) is the dim sliced
        # # but x has no enough len,so here i try slice z
        # start_slice = random.randint(0, ct_array.shape[2] - size)
        # end_slice = start_slice + size - 1
        #
        # ct_array = ct_array[ :, :,start_slice:end_slice + 1]
        # seg_array = seg_array[ :, :,start_slice:end_slice + 1]

        # 处理完毕，将array转换为tensor
        # unsqueeze(1) inserts a dimension of size one at position 1,
        # effectively changing the shape of the tensor from (3, 4) to (3, 1, 4).
        # The data is shared between the original tensor and the new tensor.
        # why unsqueeze(0) here??
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)

        return ct_array, seg_array

    def __len__(self):

        return len(self.ct_list)



ct_dir = r'C:\Git\DataSet\abdomen\train_zoomed'
seg_dir = r'C:\Git\DataSet\abdomen\label_zoomed'

train_ds = Dataset(ct_dir, seg_dir)


if __name__ == '__main__':
    # # # 测试代码
    from torch.utils.data import DataLoader
    train_ds = Dataset(ct_dir, seg_dir)
    ind = train_ds[0]
    print(ind)

    train_dl = DataLoader(train_ds, batch_size=6, shuffle=True)
    for index, (ct, seg) in enumerate(train_dl):

        print(index, ct.size(), seg.size())
        print('----------------')
