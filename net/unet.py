# import _init_paths
import torch
import torch.nn as nn
from net.layers import unetConv3, unetUp
from net.utils import init_weights, count_param
from Log import *

class UNet(nn.Module):

    def __init__(self,conv_depth, channel_factors,input_channels=1,
                 feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()

        self.conv_depth = conv_depth
        self.input_channels = input_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        channels =[n * channel_factors for n in [1,2,4,8,16,32,64,128]]

        # logger.info("2.create all the layers")
        self.maxpool = nn.MaxPool3d(kernel_size=2)

        for i in range(conv_depth):
            conv = unetConv3(input_channels, channels[i], self.is_batchnorm)
            setattr(self, 'conv%d'%i, conv)
            input_channels = channels[i]

        for i in range(conv_depth-1,-1,-1):
            if i > 0:
                ConvT_Cat_Conv = unetUp(channels[i], channels[i-1])
                setattr(self, 'ConvT_Cat_Conv%d' % i, ConvT_Cat_Conv)
            else:
                conv_out =nn.Sequential(nn.ConvTranspose3d(channels[i], channels[i], kernel_size=2, stride=2, padding=0),\
                            unetConv3(channels[i], 1, self.is_batchnorm),nn.Threshold(0.4,0,inplace=False))
                setattr(self, 'conv_out', conv_out)


        # self.modules() Returns an iterator over immediate children modules, yielding both
        # the name of the module as well as the module itself.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, input):

        layer_input = input
        for i in range(self.conv_depth):
            conv = getattr(self, 'conv%d'%i)
            conv_output = conv(layer_input)
            maxpool_output = self.maxpool(conv_output)
            # logger.info("conv{}:conv_output:{},maxpool_output:{}"
            #             .format(i,conv_output.shape,maxpool_output.shape))
            setattr(self, 'conv%d_output'%i, maxpool_output)
            layer_input = maxpool_output


        for i in range(self.conv_depth-1,-1,-1):
            if i > 0:
                ConvT_Cat_Conv = getattr(self, 'ConvT_Cat_Conv%d' % i)
                up_output = ConvT_Cat_Conv(layer_input,getattr(self,'conv%d_output'%(i-1)))
            else:
                conv_out = getattr(self, 'conv_out')
                up_output = conv_out(layer_input)

            # logger.info("up{}:up_output:{} ".format(i,up_output.shape))
            layer_input = up_output

        return layer_input

if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable

    x = Variable(torch.rand(6, 1, 16, 256, 256)).cuda()
    model = UNet(conv_depth=4,channel_factors=8).cuda()
    param = count_param(model)
    y = model(x)
    print('Output shape:', y.shape)
    print('UNet totoal parameters: %.2fM (%d)' % (param / 1e6, param))
