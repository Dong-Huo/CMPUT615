
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
from read_data import SiameseNetworkDataset, test_Video
from Config import Config
import os


def test_imshow(img0, img1, coord_vec, text=None):
    # npimg0 = img0.numpy()
    # np_img0 = npimg0.transpose(1,2,0)

    # npimg1 = img1.numpy()
    # np_img1 = npimg1.transpose(1, 2, 0)

    npimg1 =img1

    ptsX = [ int(item) for item in coord_vec[0][0:4]]
    ptsY = [ int(item) for item in coord_vec[0][4:]]

    # img_ctn0 = np_img0.copy()
    # img_ctn0 = cv2.resize(img_ctn0, (0, 0), fx=3, fy=3)
    # img_ctn0 = cv2.cvtColor(img_ctn0, cv2.COLOR_RGB2BGR)
    # cv2.imshow('show0',img_ctn0)
    # cv2.waitKey(0)
    # cv2.destroyWindow('show0')

    img_ctn1 = npimg1.copy()
    cv2.circle(img_ctn1, (ptsX[0], ptsY[0]), 3, (0,255,255), -1)
    cv2.circle(img_ctn1, (ptsX[1], ptsY[1]), 3, (0, 255, 255), -1)
    cv2.circle(img_ctn1, (ptsX[2], ptsY[2]), 3, (0, 255, 255), -1)
    cv2.circle(img_ctn1, (ptsX[3], ptsY[3]), 3, (0, 255, 255), -1)

    cv2.line(img_ctn1, (ptsX[0], ptsY[0]), (ptsX[1], ptsY[1]), (255, 0 ,0))
    cv2.line(img_ctn1, (ptsX[1], ptsY[1]), (ptsX[3], ptsY[3]), (0, 255, 0))
    cv2.line(img_ctn1, (ptsX[3], ptsY[3]), (ptsX[2], ptsY[2]), (0, 0, 255))
    cv2.line(img_ctn1, (ptsX[2], ptsY[2]), (ptsX[0], ptsY[0]), (255, 255,0))


    img_ctn1 = cv2.resize(img_ctn1,(0,0), fx= 3,fy = 3)
    # img_ctn1 = cv2.cvtColor(img_ctn1, cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_ctn1, text, (150,30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('show1',img_ctn1)
    cv2.waitKey(0)
    # cv2.destroyWindow('show1')



if __name__ == '__main__':

    net = torch.load(os.path.join(Config.test_model_dir, '75.model'))
    net.eval()

    criterion = MSELoss(l_batch_size=1,
                        reg_coord=Config.reg_coord, warp_sty=Config.warp_sty,
                        stage='test')

    test_vid_seq = test_Video(video_path = Config.testing_dir,
                    video_name= "video10",
                    transform=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])
                   )



    while(True):
        if test_vid_seq.img_index >= test_vid_seq.len_video:
            break

        org_x0, org_x1, x0, x1 = test_vid_seq.get_imgs()

        # concatenated = torch.cat((x0, x1), 0)
        x0 = torch.unsqueeze(x0, 0)
        x1 = torch.unsqueeze(x1, 0)

        cat_output = net(Variable(x0).cuda(), Variable(x1).cuda())
        target = criterion(cat_output, None)

        coord_vec = target.cpu().detach().numpy()

        test_imshow(org_x0, org_x1, coord_vec,
                    text='coords: {}'.format(target.cpu().detach().numpy()))

