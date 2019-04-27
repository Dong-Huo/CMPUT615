import numpy as np
import cv2
import os

from dlt import DLT


class generate(object):
    def __init__(self, bounding_size, source_directory, target_directory):
        self.bounding_size = bounding_size
        self.source_directory = source_directory
        self.target_directory = target_directory
        self.scale = 1
        self.trans = [0, 0]

        self.trans_range = [-20, 20]

        try:
            os.mkdir(self.target_directory)
        except:
            pass

    def generate_all(self):
        # entries = os.listdir('/Users/donghuo/Downloads/TMT-2/nl_bookI_s1')
        dirs = [x[0] for x in os.walk(self.source_directory)]
        dirs = dirs[1:]

        for item in dirs:
            if 'tags' in item:
                dirs.remove(item)
            elif 'OptGT' in item:
                dirs.remove(item)
            else:
                pass

        for i in range(len(dirs)):
            # if 'nl_juice_s5' in dirs[i]:
            self.generate_single(i, dirs[i])

    def generate_single(self, folder_id, sub_dir):

        print("Processing     " + sub_dir)

        entries = os.listdir(sub_dir)

        entries.sort()

        d = DLT(self.bounding_size)

        folder_name = sub_dir.split('/')[-1]

        try:
            os.mkdir(self.target_directory + 'trans_' + folder_name + '/')
        except:
            pass

        with open(self.target_directory + 'trans_' + folder_name + '/trans_' + folder_name + '.txt',
                  'w', encoding="UTF-8") as target:
            pass

        txt_file = sub_dir + '.txt'

        with open(txt_file, 'r', encoding="UTF-8") as source:
            lines = source.readlines()

            try:
                os.mkdir(self.target_directory + 'trans_' + folder_name + '/template_trans_' + folder_name)
            except:
                pass

            for i in range(len(lines[1:])):
                line_list = lines[i + 1].split()

                img = cv2.imread(sub_dir + '/' + line_list[0])

                line_list = [float(x) for x in line_list[1:]]

                H_template, template_window = d.get_template_window(img, line_list)

                cv2.imwrite(
                    self.target_directory + 'trans_' + folder_name + '/template_trans_' + folder_name + '/template' + str(
                        i + 1) + '.jpg',
                    template_window)

            lines = lines[1:]

            try:
                os.mkdir(
                    self.target_directory + 'trans_' + folder_name + '/search_window_trans_' + folder_name)
            except:
                pass

            for i in range(1, len(lines)):
                line_list_up = lines[i - 1].split()
                line_list_down = lines[i].split()

                img2 = cv2.imread(sub_dir + '/' + line_list_down[0])

                if img2 is None:
                    temp = line_list_down[i].split('e')

                    img2 = cv2.imread(sub_dir + '/' + temp[0] + 'e0' + temp[1])

                line_list_up = line_list_up[1:]

                line_list_up = [float(x) for x in line_list_up]

                line_list_down = line_list_down[1:]

                line_list_down = [float(x) for x in line_list_down]

                self.trans = [np.random.uniform(self.trans_range[0], self.trans_range[1], 1)[0],
                              np.random.uniform(self.trans_range[0], self.trans_range[1], 1)[0]]

                H_search, search_window, four_points = d.get_search_window(img2, line_list_up, line_list_down,
                                                                           self.scale, self.trans)

                W = d.DLT_transfer([0, 0, d.bounding_size - 1, 0, d.bounding_size - 1, d.bounding_size - 1, 0,
                                    d.bounding_size - 1],
                                   [four_points[0][0], four_points[0][1], four_points[1][0], four_points[1][1],
                                    four_points[2][0], four_points[2][1], four_points[3][0], four_points[3][1]])

                cv2.imwrite(
                    self.target_directory + 'trans_' + folder_name + '/search_window_trans_' + folder_name + '/' + str(
                        i) + '.jpg',
                    search_window)

                with open(self.target_directory + 'trans_' + folder_name + '/trans_' + folder_name + '.txt',
                          'a', encoding="UTF-8") as target:
                    target.write(str(i) + '.jpg')

                    W_inv = np.linalg.inv(W)

                    W_inv = W_inv / W_inv[2][2]

                    W = np.resize(W, (9,))
                    W_inv = np.resize(W_inv, (9,))

                    for i in range(8):
                        target.write(' ' + str(round(W[i], ndigits=6)))

                    for i in range(8):
                        target.write(' ' + str(round(W_inv[i], ndigits=6)))

                    target.write("\n")


gene = generate(224, './VOT16/', './trans_vot/')
gene.generate_all()
