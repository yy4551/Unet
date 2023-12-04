import os
import shutil
import statistics
from pad import create_path
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from scipy import stats
from data_prepare.wtf import show_me_all_your_dim

def zoom_origin_image(orimage_array,target_size):
    zoomed_image = ndimage.zoom(orimage_array,
                                (target_size[0]/orimage_array.shape[0],
                                target_size[1] / orimage_array.shape[1],
                                target_size[2] / orimage_array.shape[2]),
                                order=3)
    return zoomed_image


def zoom_all_images(orimages_path,newimages_path,target_size):
    for file_index, file in enumerate(os.listdir(orimages_path)):
        orimage = sitk.ReadImage(os.path.join(orimages_path, file), sitk.sitkInt16)
        orimage_array = sitk.GetArrayFromImage(orimage)
        zoomed_image_array = zoom_origin_image(orimage_array,target_size)
        zoomed_image = sitk.GetImageFromArray(zoomed_image_array)
        zoomed_image.SetSpacing((1,1,8))
        zoomed_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        zoomed_image.SetOrigin((0,0,0))
        sitk.WriteImage(zoomed_image, os.path.join(newimages_path, file))
    return None

if __name__ == "__main__":
    train_path = r'C:\Git\DataSet\abdomen\train_origin'
    label_path = r'C:\Git\DataSet\abdomen\label_origin'
    test_path = r'C:\Git\DataSet\abdomen\test_origin'

    zoomed_train_path = r'C:\Git\DataSet\abdomen\train_zoomed'
    zoomed_labels_path = r'C:\Git\DataSet\abdomen\label_zoomed'
    zoomed_test_path = r'C:\Git\DataSet\abdomen\test_zoomed'

    # create_path(zoomed_train_path)
    # create_path(zoomed_labels_path)
    create_path(zoomed_test_path)

    # zoom_all_images(train_path,zoomed_train_path,(16,256,256))
    # zoom_all_images(label_path, zoomed_labels_path, (16,256,256))
    zoom_all_images(test_path, zoomed_test_path, (16, 256, 256))
