import cv2
import torch
import random
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os

random.seed(2)

class SiameseNetworkDataset(Dataset):
    def __init__(self, data_root_dir_list, warp_sty, warp_choice=None,reg_coord=False,transform=None):
        self.data_root_dir_list = data_root_dir_list
        self.transform = transform

        self.video_list = [None] * len(data_root_dir_list)

        for i in range(len(data_root_dir_list)):
            self.video_list[i] = os.listdir(data_root_dir_list[i])

        # self.video_list = os.listdir(data_root_dir_list[0]) #Archive
        self.warp_choice = warp_choice
        self.reg_coord = reg_coord
        self.warp_sty = warp_sty

        len_dataset = 0
        video_list = os.listdir(data_root_dir_list[0])
        for video_name in video_list:
            video_path = os.path.join(data_root_dir_list[0], video_name)
            with open(os.path.join(video_path, video_name + '.txt'), 'r') as fp:
                content = fp.read()
                content = content.split('\n')
                content = content[:-1]
            len_dataset += len(content)
        self.len_dataset = len_dataset * len(data_root_dir_list)


    def __getitem__(self, index):

        dir_ind = random.randint(0, len(self.data_root_dir_list)-1)

        video_name = random.choice(self.video_list[dir_ind])  # video1 video2

        video_path = os.path.join(self.data_root_dir_list[dir_ind], video_name)


        with open(os.path.join(video_path, video_name + '.txt'), 'r') as fp:
            content = fp.read()
            content = content.split('\n')
            content = content[:-1]
        line = random.choice(content)       #random frame
        line = line.split(' ')

        search_wind_pth = os.path.join(video_path, 'search_window_' + video_name, line[0])

        template_pth_pre = os.path.join(video_path, 'template_' + video_name)
        template_name = random.choice(os.listdir(template_pth_pre))

        template_pth = os.path.join(video_path, 'template_' + video_name, template_name)

        search_img = cv2.imread(search_wind_pth)
        template_img = cv2.imread(template_pth)
        search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB)
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)

        Warp = [float(item) for item in line[1:9]]
        inv_Warp = [float(item) for item in line[9:]]


        if self.warp_choice == 'inv':
            warp = np.array(inv_Warp, dtype=np.float32)
        else:
            warp = np.array(Warp, dtype=np.float32)

        if self.reg_coord ==True:
            src_pts = np.array([[0, 0], [223, 0], [0, 223], [223, 223]], dtype='float32')

            mtx = np.zeros([3, 3], dtype='float32')
            mtx[2][2] = 1
            mtx[0][:] = warp[0:3]
            mtx[1][:] = warp[3:6]
            mtx[2][0:2] = warp[6:]

            src_pts = np.array([src_pts], dtype='float32')

            pointsOut = cv2.perspectiveTransform(src_pts, mtx)
            b = np.zeros(8, dtype='float32')

            for i in range(4):
                b[i] = pointsOut[0][i][0]
                b[i+4] = pointsOut[0][i][1]

            label = torch.from_numpy(b)

        else:
            if self.warp_sty=='affine':
                warp = warp[0:6]
            elif self.warp_sty == 'homo':
                warp = warp[0:]
            label = torch.from_numpy(warp)


        if self.transform is not None:
            search_img = self.transform(search_img)
            template_img = self.transform(template_img)

        return template_img, search_img, label

    def __len__(self):
        return self.len_dataset

class test_Video:
    def __init__(self, video_path, video_name,transform=None):
        self.video_path = video_path
        self.video_name = video_name
        self.transform = transform

        with open(os.path.join(self.video_path, self.video_name, self.video_name + '.txt'), 'r') as fp:
            content = fp.read()
            content = content.split('\n')
            content = content[:-1]

        self.content = content
        self.len_video = len(content)
        self.img_index = 0

    def get_template(self):
        template_pth = os.path.join(self.video_path, self.video_name, 'template_' + self.video_name, 'template.jpg')

        return

    def get_imgs(self):

        line = self.content[self.img_index]
        line = line.split(' ')

        search_wind_pth = os.path.join(self.video_path, self.video_name, 'search_window_' + self.video_name, line[0])
        template_pth = os.path.join(self.video_path, self.video_name, 'template_' + self.video_name, 'template.jpg')

        search_img = cv2.imread(search_wind_pth)
        template_img = cv2.imread(template_pth)
        search_img_transed = cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB)
        template_img_transed = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            search_img_transed = self.transform(search_img_transed)
            template_img_transed = self.transform(template_img_transed)

        self.img_index += 1

        return template_img, search_img,  template_img_transed, search_img_transed
