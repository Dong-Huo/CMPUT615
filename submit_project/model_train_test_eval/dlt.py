import numpy as np
import cv2


class DLT(object):
    def __init__(self, bounding_size):
        self.bounding_size = bounding_size

        self.length = None

    def DLT_transfer(self, ref_coor, tar_coor):
        point_number = int(len(ref_coor) / 2)

        ref_array = []

        tar_array = []

        for i in range(point_number):
            ref_array.append(np.array([ref_coor[i * 2], ref_coor[i * 2 + 1]]))
            tar_array.append(np.array([tar_coor[i * 2], tar_coor[i * 2 + 1]]))

        src_pts = np.float32(np.array(ref_array)).reshape(-1, 1, 2)
        dst_pts = np.float32(np.array(tar_array)).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, 0)

        return M

    # get the reference in size of 100 x 100
    def get_template_window(self, src, ref_coor):
        M = self.DLT_transfer(ref_coor,
                              [0, 0, self.bounding_size - 1, 0, self.bounding_size - 1, self.bounding_size - 1, 0,
                               self.bounding_size - 1])

        dst = cv2.warpPerspective(src, M, (self.bounding_size, self.bounding_size))

        return M, dst

    # get the new coordinates of 4 points based on the W, the new coordinates is relative coordinates
    def warp_next(self, W):
        a1 = np.float32(np.array([0, 0]))
        a2 = np.float32(np.array([self.bounding_size - 1, 0]))
        a3 = np.float32(np.array([self.bounding_size - 1, self.bounding_size - 1]))
        a4 = np.float32(np.array([0, self.bounding_size - 1]))

        dst = cv2.perspectiveTransform(np.array([[a1, a2, a3, a4]]), np.linalg.inv(W))

        dst -= self.original_shape[1]

        return dst

    # get the coordinates of searching box,[x,y,x,y,x,y,x,y]
    def get_bounding(self, coor, scale, trans):
        min_x = 1000000
        max_x = 0

        min_y = 1000000
        max_y = 0

        for i in range(4):
            if coor[i * 2] < min_x:
                min_x = coor[i * 2]

            if coor[i * 2] > max_x:
                max_x = coor[i * 2]

            if coor[i * 2 + 1] < min_y:
                min_y = coor[i * 2 + 1]

            if coor[i * 2 + 1] > max_y:
                max_y = coor[i * 2 + 1]

        center_x = int((min_x + max_x) // 2)
        center_y = int((min_y + max_y) // 2)

        minimal_width = int(max_x - min_x)
        minimal_height = int(max_y - min_y)

        minimal_size = minimal_width
        if minimal_height > minimal_width:
            minimal_size = minimal_height

        if self.length is None:
            self.length = minimal_size

        center_x += trans[0]
        center_y += trans[1]

        minimal_size = minimal_size * scale

        top_left = np.array([center_x - minimal_size, center_y - minimal_size])
        top_right = np.array([center_x + minimal_size, center_y - minimal_size])
        bottom_right = np.array([center_x + minimal_size, center_y + minimal_size])
        bottom_left = np.array([center_x - minimal_size, center_y + minimal_size])

        return np.array([top_left, top_right, bottom_right, bottom_left])

    def get_search_window(self, src, up_coor, down_coor, scale, trans):

        unwarp_corner = self.get_bounding(up_coor, scale, trans)

        r_average = np.mean(src[:, :, 0])

        g_average = np.mean(src[:, :, 1])

        b_average = np.mean(src[:, :, 2])

        original_shape = np.shape(src)

        self.original_shape = original_shape

        padding = cv2.copyMakeBorder(src, original_shape[1], original_shape[1], original_shape[1], original_shape[1],
                                     cv2.BORDER_CONSTANT, value=[r_average, g_average, b_average])

        top_y = unwarp_corner[0][1] + original_shape[1]
        bottom_y = unwarp_corner[2][1] + original_shape[1]

        left_x = unwarp_corner[0][0] + original_shape[1]
        right_x = unwarp_corner[2][0] + original_shape[1]

        # crop = padding[top_y:bottom_y, left_x:right_x]

        M = self.DLT_transfer(
            [unwarp_corner[0][0] + original_shape[1], unwarp_corner[0][1] + original_shape[1],
             unwarp_corner[1][0] + original_shape[1] - 1, unwarp_corner[1][1] + original_shape[1],
             unwarp_corner[2][0] + original_shape[1] - 1, unwarp_corner[2][1] + original_shape[1] - 1,
             unwarp_corner[3][0] + original_shape[1], unwarp_corner[3][1] + original_shape[1] - 1],
            [0, 0, self.bounding_size - 1, 0, self.bounding_size - 1, self.bounding_size - 1, 0, self.bounding_size - 1])

        dst = cv2.warpPerspective(padding, M, (self.bounding_size, self.bounding_size))

        four_points = []

        for i in range(4):
            temp = np.dot(M,
                          np.array([down_coor[i * 2] + original_shape[1], down_coor[i * 2 + 1] + original_shape[1], 1]))

            four_points.append(np.array([temp[0] / temp[2], temp[1] / temp[2]]))

        return M, dst, np.array(four_points)

    def change_range(self, value, max_length):
        return (value / max_length * 2 - 1)

    # get the absolute coordinates based on the bounding box,
    # np.array([np.array([x,y]),np.array([x,y]),np.array([x,y]),np.array([x,y])])
    def get_abs_coor(self, M, four_points):
        absolute_coor = np.zeros([4, 2])

        for i in range(4):
            temp = np.dot(np.linalg.inv(M), np.array([four_points[i][0], four_points[i][1], 1]))

            absolute_coor[i][0] = temp[0] / temp[2] - self.original_shape[1]
            absolute_coor[i][1] = temp[1] / temp[2] - self.original_shape[1]

        return absolute_coor

    def get_next_window(self, src, absolute_coor, scale, trans):

        unwarp_corner = self.get_bounding(
            [absolute_coor[0][0], absolute_coor[0][1], absolute_coor[1][0], absolute_coor[1][1], absolute_coor[2][0],
             absolute_coor[2][1], absolute_coor[3][0], absolute_coor[3][1]], scale, trans)

        r_average = np.mean(src[:, :, 0])

        g_average = np.mean(src[:, :, 1])

        b_average = np.mean(src[:, :, 2])

        original_shape = np.shape(src)

        self.original_shape = original_shape

        padding = cv2.copyMakeBorder(src, original_shape[1], original_shape[1], original_shape[1], original_shape[1],
                                     cv2.BORDER_CONSTANT, value=[r_average, g_average, b_average])

        top_y = unwarp_corner[0][1] + original_shape[1]
        bottom_y = unwarp_corner[2][1] + original_shape[1]

        left_x = unwarp_corner[0][0] + original_shape[1]
        right_x = unwarp_corner[2][0] + original_shape[1]

        # crop = padding[top_y:bottom_y, left_x:right_x]

        M = self.DLT_transfer(
            [unwarp_corner[0][0] + original_shape[1], unwarp_corner[0][1] + original_shape[1],
             unwarp_corner[1][0] + original_shape[1] - 1, unwarp_corner[1][1] + original_shape[1],
             unwarp_corner[2][0] + original_shape[1] - 1, unwarp_corner[2][1] + original_shape[1] - 1,
             unwarp_corner[3][0] + original_shape[1], unwarp_corner[3][1] + original_shape[1] - 1],
            [0, 0, self.bounding_size - 1, 0, self.bounding_size - 1, self.bounding_size - 1, 0, self.bounding_size - 1])

        dst = cv2.warpPerspective(padding, M, (self.bounding_size, self.bounding_size))

        return M, dst

    # nl_bookI_s1
    # def pipeline(self):
    #     img = cv2.imread("frame00001.jpg")
    #     # warp template to 224 x 224, and return the H matrix and output image
    #     H_template, template_window = self.get_template_window(img,
    #                                                            [308.00, 310.00, 442.00, 302.00, 449.00, 504.00, 320.00,
    #                                                             510.00])
    #
    #     # cv2.imshow("hh", template_window)
    #     # cv2.waitKey(0)
    #
    #     # crop the bounding box from image and warp to 224 x 224,
    #     # return H matrix, output image, and 4 coordinates of points after warp
    #     img2 = cv2.imread("frame00002.jpg")
    #     H_search, search_window, four_points = self.get_search_window(img2,
    #                                                                   [365.59, 442.65, 416.51, 442.85, 416.32, 491.34,
    #                                                                    365.40, 491.13],1)
    #
    #     # You train a model and get a warp matrix W, assume it is identity matrix
    #     W = np.array([])
    #
    #     relative_coor = self.warp_next(W)
    #
    #     absolute_coor = self.get_abs_coor(relative_coor, H_search)
    #
    #     print(absolute_coor)

# d = DLT(224)
# d.pipeline()
