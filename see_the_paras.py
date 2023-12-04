import torch

# Load the pretrained model
model_path = r'C:\Git\NeuralNetwork\abdominal-multi-organ-segmentation\module\net5-0.943-0.959.pth'
pretrained_model = torch.load(model_path)

# Access and print the model parameters
for name, param in pretrained_model.items():
    print(f"Parameter name: {name}, Shape: {param.shape}")