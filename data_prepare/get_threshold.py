"""
获取合理的阈值，将原始数据的灰度值截断到一定的范围内
减少无关数据的影响
实验记录在下方，综合实验结果，最终选择±350
"""

import os
import SimpleITK as sitk


ct_path = r'C:\Git\DataSet\abdomen\averaged-training-images'
seg_path = r'C:\Git\DataSet\abdomen\averaged-training-labels'
# seg_path = '/home/zcy/Desktop/dataset/multi-organ/test/GT'

# 定于阈值
upper = 350
lower = -350

num_point = 0.0  # 属于器官的体素数量
num_inlier = 0.0  # 器官体素中灰度值位于阈值之内的体素数量

for ct_file in os.listdir(ct_path):

    # 将原始ct数据和对应金标准读入到内存中
    ct = sitk.ReadImage(os.path.join(ct_path, ct_file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(seg_path, ct_file.replace('avg', 'avg_seg')), sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # True in seg as a whole is ROI
    # 选取属于ROI的体素
    organ_roi = ct_array[seg_array > 0]

    # This part creates a boolean mask where each element is True if
    # the corresponding voxel value in organ_roi is within the specified range (lower to upper).
    # The * operator performs element-wise multiplication of the two boolean arrays.

    #.astype(int): This converts the boolean array to an integer array,
    # where True is represented as 1 and False as 0.

    #.sum(): Finally, this calculates the sum of the integer array,
    # giving the total number of inliers within the specified range.

    #use upper&lower to crop ROI pixel-wise,all the lefted is inliers
    inliers = ((organ_roi <= upper) * (organ_roi >= lower)).astype(int).sum()

    # inliers/ROI = percentage of remain of ROI
    print('one:{:.4}%'.format(inliers / organ_roi.shape[0] * 100))
    print('------------')

    num_point += organ_roi.shape[0]
    num_inlier += inliers

print(f"two:{num_inlier / num_point}")

# ±200的阈值对于肝脏：训练集93.8%
# ±250的阈值对于肿瘤：训练集96.3%
# ±300的阈值对于肿瘤：训练集97.4%， 测试集98.9%
# ±350的阈值对于肿瘤：训练集97.7%， 测试集99%
# ±400的阈值对于肿瘤：训练集97.8%， 测试集99%
