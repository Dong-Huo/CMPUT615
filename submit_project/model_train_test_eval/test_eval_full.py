
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
from data_stuff import data_stuff
from shapely.geometry import Polygon
from time import time

def test_SW_imshow(img0, img1, coord_vec, delay=1, text=None):
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


    img_ctn1 = cv2.resize(img_ctn1,(0,0), fx= 2,fy = 2)
    # img_ctn1 = cv2.cvtColor(img_ctn1, cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_ctn1, text, (150,30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Search window',img_ctn1)
    cv2.waitKey(delay)
    # cv2.destroyWindow('show1')

def test_imshow(img0, img1, coords, delay=33, text=None):
    # npimg0 = img0.numpy()
    # np_img0 = npimg0.transpose(1,2,0)

    # npimg1 = img1.numpy()
    # np_img1 = npimg1.transpose(1, 2, 0)

    coords = coords.astype(int)

    npimg1 =img1
    ptsX = [None]*4
    ptsY = [None]*4

    ptsX[0], ptsX[1], ptsX[3], ptsX[2] = coords[:, 0]
    ptsY[0], ptsY[1], ptsY[3], ptsY[2] = coords[:, 1]

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


    img_ctn1 = cv2.resize(img_ctn1,(0,0), fx= 2,fy = 2)
    # img_ctn1 = cv2.cvtColor(img_ctn1, cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_ctn1, text, (150,30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('video',img_ctn1)
    cv2.waitKey(delay)
    # cv2.destroyWindow('show1')

def polygon_area(x,y):
    # coordinate shift
    x_ = x - x.mean()
    y_ = y - y.mean()
    # everything else is the same as maxb's code
    correction = x_[-1] * y_[0] - y_[-1]* x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5*np.abs(main_area + correction)


if __name__ == '__main__':

    net = torch.load(os.path.join(Config.test_model_dir, '100.model'))
    net.eval()

    criterion = MSELoss(l_batch_size=1,
                        reg_coord=Config.reg_coord, warp_sty=Config.warp_sty,
                        stage='test')

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                    ])

    # dir_list = Config.test_folder
    # dir_list = Config.train_folder
    dir_list = Config.spec_folder
    IOU_record = []
    for item in  dir_list:

        # df = data_stuff('./data/VOT14/' + item)  #gril
        df = data_stuff('./data/VOT16/' + item)
        absolute_coor = np.zeros(shape=[4, 2])

        area_list = []
        sub = 0

        first_time_runing = True

        while(True):
            load_time = time()

            template = df.get_template()

            search_window, org_frame = df.get_image(absolute_coor.astype(int))
            # cv2.imshow('search',search_window)
            # cv2.waitKey(300)

            template_img_transed = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
            search_img_transed = cv2.cvtColor(search_window, cv2.COLOR_BGR2RGB)

            template_img_transed = transform(template_img_transed)
            search_img_transed = transform(search_img_transed)

            template_img_transed = torch.unsqueeze(template_img_transed , 0)
            search_img_transed = torch.unsqueeze(search_img_transed, 0)

            #print('load time = {}'.format(time()-load_time))
            net_time = time()

            cat_output = net(Variable(template_img_transed).cuda(), Variable(search_img_transed).cuda())
            target = criterion(cat_output, None)

            # print('net time = {}'.format(time() - net_time))

            trans_abs_time = time()
            coord_vec = target.cpu().detach().numpy()
            ptsX = [int(item) for item in coord_vec[0][0:4]]
            ptsY = [int(item) for item in coord_vec[0][4:]]

            four_points = np.array([[ptsX[0], ptsY[0]],[ptsX[1], ptsY[1]],[ptsX[3], ptsY[3]],[ptsX[2], ptsY[2]]])
            absolute_coor = df.get_absolute_coor(four_points)

            # print('trans abs time = {}'.format(time() - trans_abs_time))

            # this is for avoiding overflow when fail in tracking
            absolute_coor[absolute_coor > 2*max(org_frame.shape[0:2])] = 2*max(org_frame.shape[0:2])
            absolute_coor[absolute_coor < -2 * max(org_frame.shape[0:2])] = - 2 * max(org_frame.shape[0:2])

            poly=Polygon(absolute_coor)
            area_list.append(poly.area)

            if len(area_list)>1:
                sub = area_list[-1] - area_list[-2]
            else:
                sub = 0

            try:
                if first_time_runing:
                    delay = 0
                    first_time_runing = False
                else:
                    delay = 33
                test_SW_imshow(None, search_window, coord_vec, 1)

                test_imshow(template, org_frame, absolute_coor, delay, text=None)
            except:
                pass


            if df.is_done():
                break
        IOU_vid = df.get_IOU()
        print(item, ' = ', IOU_vid)
        IOU_record.append(IOU_vid)
    print('mean: {}'.format(np.mean(IOU_record)))
    input()