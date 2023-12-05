from data_prepare import *
import os
import torch
import numpy as np
import SimpleITK as sitk
from net.unet import UNet
from data_prepare.pad import create_path

def predict_img(net, test_image_path, out_threshold=0.5):
    test_image = sitk.ReadImage(test_image_path)
    test_array = sitk.GetArrayFromImage(test_image)
    test_tensor = torch.FloatTensor(test_array).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
    test_tensor = test_tensor.to(device="cuda", dtype=torch.float32)

    with torch.no_grad():
        output = net(test_tensor).cpu()
        output = torch.sigmoid(output) > out_threshold
        output = output.squeeze().numpy().astype(float)

        predict_image = sitk.GetImageFromArray(output)
        predicted_image_mask = sitk.BinaryThreshold(predict_image, lowerThreshold=0.5)
        return predicted_image_mask

def predict_all(net, test_path, module_path, new_path):
    net.load_state_dict(torch.load(module_path))
    net.eval()
    for file in os.listdir(test_path):
        test_image_path = os.path.join(test_path, file)
        predicted_image_mask = predict_img(net, test_image_path)
        predicted_image_mask.SetOrigin((0, 0, 0))
        predicted_image_mask.SetSpacing((1,1,8))
        predicted_image_mask.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        sitk.WriteImage(predicted_image_mask, os.path.join(new_path, file))




if __name__ == "__main__":
    net = UNet(conv_depth=4,channel_factors=8).cuda()
    module_path = r"C:\Git\NeuralNetwork\Unet\pth\net100-0.222-0.232.pth"
    test_path = r"C:\Git\DataSet\abdomen\test_zoomed"
    new_path = r"C:\Git\DataSet\abdomen\test_predicted"
    create_path(new_path)

    predict_all(net, test_path, module_path,new_path)
