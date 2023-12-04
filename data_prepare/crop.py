import os
import shutil

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

def find_min_dimensions(iamges_paths):
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')

    for ct_file in os.listdir(iamges_paths):
        # 读取 NIfTI 文件
        image = sitk.ReadImage(os.path.join(iamges_paths, ct_file), sitk.sitkInt16)
        image_array = sitk.GetArrayFromImage(image)

        # 获取图像尺寸
        x_size, y_size, z_size = image_array.shape

        # 更新最小尺寸
        min_x = min(min_x, x_size)
        min_y = min(min_y, y_size)
        min_z = min(min_z, z_size)

    return min_x, min_y, min_z


def find_max_dimensions(iamges_paths):
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

    for ct_file in os.listdir(iamges_paths):
        # 读取 NIfTI 文件
        image = sitk.ReadImage(os.path.join(iamges_paths, ct_file), sitk.sitkInt16)
        image_array = sitk.GetArrayFromImage(image)

        # 获取图像尺寸
        x_size, y_size, z_size = image_array.shape

        # 更新最小尺寸
        max_x = max(max_x, x_size)
        max_y = max(max_y, y_size)
        max_z = max(max_z, z_size)

    return max_x, max_y, max_z

def crop_images(origional_paths,new_paths):
    # 找到最小尺寸
    min_x, min_y, min_z = find_min_dimensions(origional_paths)

    minyz = min(min_z,min_y)

    for file_index,ct_file in enumerate(os.listdir(origional_paths)):
        # 读取 NIfTI 文件
        ct = sitk.ReadImage(os.path.join(origional_paths, ct_file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        # 裁剪图像
        ct_array = ct_array[:min_x, :minyz, :minyz]

        # 保存裁剪后的图像
        new_ct = sitk.GetImageFromArray(ct_array)
        sitk.WriteImage(new_ct, os.path.join(new_paths, ct_file))



if __name__ == "__main__":
    # 指定图像文件目录和输出目录
    ct_path = r'C:\Git\DataSet\abdomen\averaged_training_images'
    seg_path = r'C:\Git\DataSet\abdomen\averaged_training_labels'

    new_ct_path = r'C:\Git\DataSet\abdomen\train_crop'
    new_seg_path = r'C:\Git\DataSet\abdomen\label_crop'

    if os.path.exists(r'C:\Git\DataSet\abdomen\train_crop') is True:
        shutil.rmtree(r'C:\Git\DataSet\abdomen\train_crop')
    os.mkdir(r'C:\Git\DataSet\abdomen\train_crop')

    if os.path.exists(r'C:\Git\DataSet\abdomen\label_crop') is True:
        shutil.rmtree(r'C:\Git\DataSet\abdomen\label_crop')
    os.mkdir(r'C:\Git\DataSet\abdomen\label_crop')


    crop_images(ct_path,new_ct_path)
    crop_images(seg_path,new_seg_path)

