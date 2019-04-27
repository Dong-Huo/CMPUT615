# %matplotlib inline
import torchvision
from torchvision import models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
# import matplotlib
# matplotlib.use('asd')
# import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import cv2
from siamese_module import SiameseNetwork, MSELoss
from read_data import SiameseNetworkDataset
from Config import Config
import os

def vis_show(img,text=None):
    npimg = img.numpy()
    np_img = npimg.transpose(1,2,0)

    font = cv2.FONT_HERSHEY_SIMPLEX

    img_from_container = np_img.copy()
    img_from_container = cv2.resize(img_from_container,(0,0), fx= 3,fy = 3)
    img_from_container = cv2.cvtColor(img_from_container, cv2.COLOR_RGB2BGR)
    cv2.putText(img_from_container, text, (150,30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('show',img_from_container)
    cv2.waitKey(0)
    cv2.destroyWindow('show')

def test_imshow(img,cat_output, text=None):
    npimg = img.numpy()
    np_img = npimg.transpose(1,2,0)

    mtx = np.zeros([3, 3], dtype=float)
    mtx[2][2] = 1

    mtx[0][:] = cat_output[0][0:3]
    mtx[1][:] = cat_output[0][3:6]
    mtx[2][0:2] = cat_output[0][6:]
    print(mtx)


    src_pts = np.array([[0,0],[0,223],[223,0],[223,223]])
    cv2.perspectiveTransform(src_pts, )

    font = cv2.FONT_HERSHEY_SIMPLEX

    img_from_container = np_img.copy()
    img_from_container = cv2.resize(img_from_container,(0,0), fx= 3,fy = 3)
    img_from_container = cv2.cvtColor(img_from_container, cv2.COLOR_RGB2BGR)
    cv2.putText(img_from_container, text, (150,30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('show',img_from_container)
    cv2.waitKey(0)
    cv2.destroyWindow('show')


if __name__ == '__main__':


    siamese_dataset = SiameseNetworkDataset(data_root_dir_list = Config.training_dir_list,
                                            warp_sty = Config.warp_sty,
                                            warp_choice = Config.warp_choice,
                                            reg_coord=True,
                                            transform=transforms.Compose([transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                                        ])
                                           )

    vis_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=4,
                            batch_size=2)

    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0],example_batch[1]),2)
    print(example_batch[2].numpy())
    vis_show(torchvision.utils.make_grid(concatenated))



    train_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=4,
                            batch_size=Config.train_batch_size)


    # VGG11bn_model = models.vgg11_bn(pretrained=True)
    alexnet_model = models.alexnet(pretrained=True)
    for param in alexnet_model.parameters():
        param.requires_grad = False


    num_mtx_params = None
    if Config.warp_sty == 'homo':
        num_mtx_params = 8
    elif Config.warp_sty == 'affine':
        num_mtx_params = 6
    elif Config.warp_sty == 'simi':
        num_mtx_params = 4

    net = SiameseNetwork(backbone= alexnet_model, num_mtx_params= num_mtx_params).cuda()

    print(net)

    # net = torch.load(os.path.join('./model_alex/', '50.model'))
    # net.eval()

    # for param in net.parameters():
    #     print(param.requires_grad)

    feature_extract = False
    print("Params to learn:")

    params_to_update = net.parameters()
    if feature_extract:
        params_to_update = []
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # criterion = MSELoss(l_batch_size = Config.train_batch_size,
    #                     reg_coord=Config.reg_coord, warp_sty = Config.warp_sty,
    #                     stage='train')

    criterion = MSELoss(l_batch_size = Config.train_batch_size,
                        reg_coord=False, warp_sty = Config.warp_sty,
                        stage='train')

    optimizer = optim.Adam(params_to_update,lr = Config.lr)
    # optimizer = optim.SGD(params_to_update,lr = 0.0005 )

    counter = []
    loss_history = []
    iteration_number= 0


    for epoch in range(0,Config.train_number_epochs+1):

        for i, data in enumerate(train_dataloader,0):
            template_img, search_img , label = data
            template_img, search_img , label = template_img.cuda(), search_img.cuda() , label.cuda()
            optimizer.zero_grad()
            cat_output = net(template_img,search_img)
            loss_func = criterion(cat_output, label)
            loss_func.backward()
            optimizer.step()
            if i % 200 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_func.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_func.item())
        if epoch %10 ==0:
            torch.save(net, './model/{}.model'.format(epoch))

