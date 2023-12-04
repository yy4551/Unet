import os
from time import time
import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from data_prepare.pad import create_path
from Log import logger
from net.unet import UNet
from loss.Dice_loss import myDiceLoss
from dataset.dataset import train_ds
output_path = r'C:\Git\DataSet\abdomen\train_outcome'
def train():

    for epoch in range(Epoch):
        print('-------------epoch {}-------------------'.format(epoch + 1))
        start = time()
        mean_loss = []

        for step, (train_data, label_data) in enumerate(train_dl):

            train_data = train_data.to("cuda")
            label_data = label_data.to("cuda")

            predict= net(train_data)

            loss = loss_func(predict, label_data)
            mean_loss.append(loss.item())

            opt.zero_grad()
            loss.backward() # When you call loss.backward(), it computes the gradient of the loss with respect to the model parameters. These gradients are stored in the .grad attribute of the respective parameters.
            opt.step() # The opt.step() function is called after loss.backward() to use these gradients to update the parameters. The specific way in which the parameters are updated depends on the optimization algorithm. In your case, you're using the Adam optimizer, so the parameters are updated according to the Adam optimization algorithm.
            lr_decay.step()

            if step % 8 == 0:
                print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                      .format(epoch, step, loss.item(), (time() - start) / 60))
                logger.info('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                      .format(epoch, step, loss.item(), (time() - start) / 60))
            pass

        mean_loss = sum(mean_loss) / len(mean_loss)

        # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
        if epoch % 25 == 0:
            torch.save(net.state_dict(), './pth/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss.item(), mean_loss))


if __name__ == '__main__':
    Epoch = 100
    leaing_rate = 0.001
    batch_size = 21

    logger.info("epoch:{},learning rate:{},batch size:{}"
                .format(Epoch, leaing_rate, batch_size))

    net = UNet(conv_depth=4,channel_factors=8).to("cuda")
    logger.info("Unet qualified")

    train_dl = DataLoader(train_ds, batch_size, True, num_workers=2)
    logger.info("dataloader ready")

    loss_func = myDiceLoss()
    logger.info("loss function ready")

    opt = torch.optim.Adam(net.parameters(), lr=leaing_rate)
    print("optimizer initialized")

    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [80])
    print("lr decay initialized")

    logger.info("------------start training!----------------")

    train()
