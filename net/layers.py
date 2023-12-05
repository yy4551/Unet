import torch
import torch.nn as nn
from net.utils import init_weights
import torch.nn.functional as F

# ks = kernel_size

# __init__ is executed when an instance is created,namely,net = Unet()
# forward is  executed when an instance is called,namely,during training
class unetConv3(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv3, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding

        if is_batchnorm:
            # each conv has n (Conv3d+ReLU+BN) blocks
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, ks, s, p),
                                     nn.BatchNorm3d(out_channels),
                                     nn.ReLU(inplace=True),)

                setattr(self, 'conv%d'%i, conv)
                in_channels = out_channels
        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_channels = out_channels

        # conv1,conv2,conv3... are the child modules
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            # conv here is from the __init__ above
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)
        return x

class unetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # attention:it's maxpooling makes tentor smaller,ks of maxpooling is 2
        # now want to make it bigger,so use ConvTranspose3d,ks of convT is 2 same as maxpooling
        self.convT3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = unetConv3(in_channels, out_channels, False)

        # init weights for all filters in this class?
        for m in self.children():
            if m.__class__.__name__.find('unetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, low_feature):
        # convT3d and conv from __init__ above
        output = self.convT3d(high_feature)
        output = torch.cat([output, low_feature], 1)
        return self.conv(output)

if __name__ == "__main__":



