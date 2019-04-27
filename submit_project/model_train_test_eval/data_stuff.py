import cv2
import numpy as np

from dlt import DLT
from shapely.geometry import Polygon


class data_stuff(object):

    def __init__(self, path):
        self.path = path
        self.d_function = DLT(224)
        self.search_index = 0
        self.scale = 1
        self.trans = [0, 0]

        self.current_ground_truth = None
        self.frame_num = 0

        self.initialize()

        self.IOU = 0
        # self.max_length = 0

    def initialize(self):
        with open(self.path + '.txt', 'r', encoding="UTF-8") as source:
            lines = source.readlines()
            line_list = lines[1].split()

            self.frame_num = len(lines) - 3

            img = cv2.imread(self.path + '/' + line_list[0])

            line_list = line_list[1:]

            self.max_length = len(lines)

            line_list = [float(x) for x in line_list]

            M, dst = self.d_function.get_template_window(img, line_list)

            line_list_down = lines[2].split()

            img2 = cv2.imread(self.path + '/' + line_list_down[0])

            line_list_down = line_list_down[1:]
            line_list_down = [float(x) for x in line_list_down]

            H_search, search_window, four_points = self.d_function.get_search_window(img2, line_list, line_list_down,
                                                                                     self.scale, self.trans)

            self.four_points = four_points

            self.warp_matrix = H_search

            self.template_windows = dst
            self.search_window = search_window
            self.img = img2

            self.current_ground_truth = line_list_down

            # return dst, search_window

    def get_template(self):
        return self.template_windows

    def get_image(self, absolute_coor):

        # print(self.search_index)

        if self.search_index == 0:
            self.search_index += 1
            return self.search_window, self.img

        with open(self.path + '.txt', 'r', encoding="UTF-8") as source:
            lines = source.readlines()
            lines = lines[1:]

            line_list = lines[self.search_index + 1].split()

            img = cv2.imread(self.path + '/' + line_list[0])

            H_search, search_window = self.d_function.get_next_window(img, absolute_coor, self.scale, self.trans)

            line_list = [float(x) for x in line_list[1:]]
            #
            # _, _, self.four_points = self.d_function.get_search_window(img, [absolute_coor[0][0], absolute_coor[0][1],
            #                                                                  absolute_coor[1][0], absolute_coor[1][1],
            #                                                                  absolute_coor[2][0], absolute_coor[2][1],
            #                                                                  absolute_coor[3][0], absolute_coor[3][1]]
            #                                                                  ,line_list, self.scale, self.trans)

            self.current_ground_truth = line_list

            self.search_index += 1

            self.warp_matrix = H_search

            return search_window, img

    def set_scale(self, scale):
        self.scale = scale

    def get_absolute_coor(self, four_points):
        absolute_coor = self.d_function.get_abs_coor(self.warp_matrix, four_points)

        self.IOU += self.calculate_IOU(absolute_coor)

        return absolute_coor

    def is_done(self):
        if self.search_index == self.max_length - 3:
            return True
        else:
            return False

    def calculate_IOU(self, absolute_coor):

        p1 = Polygon([(absolute_coor[0][0], absolute_coor[0][1]), (absolute_coor[1][0], absolute_coor[1][1]),
                      (absolute_coor[2][0], absolute_coor[2][1]), (absolute_coor[3][0], absolute_coor[3][1])])
        p2 = Polygon([(self.current_ground_truth[0], self.current_ground_truth[1]),
                      (self.current_ground_truth[2], self.current_ground_truth[3]),
                      (self.current_ground_truth[4], self.current_ground_truth[5]),
                      (self.current_ground_truth[6], self.current_ground_truth[7])])
        p3 = p1.intersection(p2)
        # print(p3.area)

        # print(p3.area / (p1.area + p2.area - p3.area))

        return p3.area / (p1.area + p2.area - p3.area)

    def get_IOU(self):

        self.IOU /= self.frame_num

        return self.IOU


