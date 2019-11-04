import os
import cv2
import numpy as np

original_dir = '../test_data/image/'
all_res_dir = '../all_3d_tf_res/'
save_dir = ''

original_im_name = 'Cameraman.png'

im = cv2.imread(original_dir + original_im_name, cv2.IMREAD_GRAYSCALE)

for im_name in os.listdir(all_res_dir):
    if original_im_name[:-4] in im_name:
        res_im = cv2.imread(all_res_dir+im_name, cv2.IMREAD_GRAYSCALE)
        diff_im = np.abs(res_im.astype(np.int)-im.astype(np.int))
        diff_im = diff_im.astype(np.uint8)
        cv2.imwrite(save_dir + im_name, diff_im)
