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
from dataset.dataset import train_ds,output_tensor_label,output_tensor_pred
import h5py
from torch.autograd import Variable

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

            # output_tensor_label.add_tensor(label_data)
            # output_tensor_pred.add_tensor(predict)
            # if len(output_tensor_label) >= 20:
            #     torch.save(output_tensor_label.tensor_list, r".\output_tensor_label_epoch%d.pt"%epoch)
            #     torch.save(output_tensor_pred.tensor_list, r".\output_tensor_pred_epoch%d.pt"%epoch)
            #     output_tensor_label.tensor_list = []
            #     output_tensor_pred.tensor_list = []

            logger.info("********************************************************")
            loss = loss_func(predict, label_data)
            logger.info("--------------epoch:{}, step:{}--------------".format(epoch+1, step+1))
            logger.info("********************************************************")
            if loss < 0:
                with h5py.File('negative_loss.h5', 'w') as hf:
                    hf.create_dataset("negative_loss", data=predict.detach().cpu().squeeze().numpy().astype(float))
                raise ValueError("Hell,your loss is negative again,come and check it out!")

            mean_loss.append(loss.item())

            opt.zero_grad()
            loss.backward() # When you call loss.backward(), it computes the gradient of the loss with respect to the model parameters. These gradients are stored in the .grad attribute of the respective parameters.
            opt.step() # The opt.step() function is called after loss.backward() to use these gradients to update the parameters. The specific way in which the parameters are updated depends on the optimization algorithm. In your case, you're using the Adam optimizer, so the parameters are updated according to the Adam optimization algorithm.
            lr_decay.step()

            if step % 3 == 0:
                print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                      .format(epoch+1, step+1, loss.item(), (time() - start) / 60))
                # logger.info('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'.format(epoch, step, loss.item(), (time() - start) / 60))
            pass

        mean_loss = sum(mean_loss) / len(mean_loss)

        # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
        if (epoch+1) % 25 == 0:
            torch.save(net.state_dict(), './pth/net{}-{:.3f}-{:.3f}.pth'.format(epoch+1, loss.item(), mean_loss))


if __name__ == '__main__':

    Epoch = 100
    leaing_rate = 0.001
    batch_size = 20

    negative_loss_path = r".\negative_loss.h5"

    logger.info("epoch:{},learning rate:{},batch size:{}"
                .format(Epoch, leaing_rate, batch_size))

    net = UNet(conv_depth=4,channel_factors=8).to("cuda")
    logger.info("Unet qualified")

    train_dl = DataLoader(train_ds, batch_size, True, num_workers=2)
    logger.info("dataloader ready")

    loss_func = myDiceLoss()
    logger.info("loss function ready")

    opt = torch.optim.Adam(net.parameters(), lr=leaing_rate)

    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [80])

    logger.info("*******************************************"
                "------------start training!---------------"
                "*******************************************")

    train()
