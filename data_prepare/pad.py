import os
import shutil
import statistics

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from scipy import stats


def determine_size(ndim, path):
    dim_list = []
    single_dim = []
    dim_size = []

    for dim in range(ndim):
        for ct_file in os.listdir(path):
            array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, ct_file), sitk.sitkInt16))
            single_dim.append(array.shape[dim])

        # dim_list.append(list(map(float, single_dim)))
        dim_list.append(single_dim)
        if statistics.mode(dim_list[dim]) >= 16:
            mode = statistics.mode(dim_list[dim])
        else:
            mode = 16
        dim_size.append(mode)

    return dim_size


def padding(array, dim, dim_target_size):
    if dim == 0:pad_width = ((0, dim_target_size - array.shape[dim]),(0,0),(0,0))
    if dim == 1:pad_width = ( (0, 0),(0, dim_target_size - array.shape[dim]),(0, 0))
    if dim == 2:pad_width = ((0, 0), (0, 0), (0, dim_target_size - array.shape[dim]))

    padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)
    return padded_array

def cropping(array, dim, dim_target_size):
    # array = np.array([sub[:dim_target_size] if sub is array[dim] else sub for sub in array])
    if dim == 0:
        array = array[:dim_target_size,:,:]
    if dim == 1:
        array = array[:,:dim_target_size,:]
    if dim == 2:
        array = array[:,:,:dim_target_size]
    return array

def crop_and_pad(origin_path, new_path, ndim):
    target_size =[16,256,256]

    for file_index, file in enumerate(os.listdir(origin_path)):
        array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(origin_path, file), sitk.sitkInt16))

        for dim in range(ndim):
            if array.shape[dim] < target_size[dim]:
                array = padding(array, dim, target_size[dim])
            if array.shape[dim] > target_size[dim]:
                array = cropping(array, dim, target_size[dim])

        new_image = sitk.GetImageFromArray(array)

        if origin_path.find('train') != -1:
            new_name = 'img_' + str(file_index) + '.nii'
        if origin_path.find('label') != -1:
            new_name = 'label-' + str(file_index) + '.nii'
        if origin_path.find('test') != -1:
            new_name = 'test_' + str(file_index) + '.nii'


        print(f"{new_name} dim = {array.shape},spacing = {new_image.GetSpacing()}")

        sitk.WriteImage(new_image, os.path.join(new_path, new_name))


def set_spacing(path, spacing):
    for file in os.listdir(path):
        ct = sitk.ReadImage(os.path.join(path, file), sitk.sitkInt16)
        ct.SetSpacing(spacing)

def create_path(path):
    if os.path.exists(path) is True:
        shutil.rmtree(path)
    os.mkdir(path)
    return path

if __name__ == "__main__":
    origin_ct_path = r'C:\Git\DataSet\abdomen\train_origin'
    origin_seg_path = r'C:\Git\DataSet\abdomen\label_origin'
    origin_test_path = r'C:\Git\DataSet\abdomen\test_origin'

    crop_and_pad_path_train = r'C:\Git\DataSet\abdomen\train_crop_and_pad'
    crop_and_pad_path_label = r'C:\Git\DataSet\abdomen\label_crop_and_pad'
    crop_and_pad_path_test = r'C:\Git\DataSet\abdomen\test_crop_and_pad'

    #
    # crop_and_pad(origin_ct_path, create_path(crop_and_pad_path_train),3)
    # crop_and_pad(origin_seg_path, create_path(crop_and_pad_path_label), 3)
    # set_spacing(crop_and_pad_path_train, (1.5,1.5,8))
    # set_spacing(crop_and_pad_path_label, (1.5,1.5,8))
    crop_and_pad(origin_test_path, create_path(crop_and_pad_path_test), 3)



