import os
import cv2
import numpy as np
from GEAR.utils import dict_product
from GEAR.res_name_dencode import name_hyper_2_str, dict2name, is_contain_all, decode_res
import matplotlib.pyplot as plt

search_dir = '../all_3d_tf_res'

im_na = 'Cameraman'

hyper_dict = {
    'sigma': [20],
    'kW': [4, 5, 6, 7, 8, 9, 10, 11, 12],
    'nW': [0],
    'NW': [8, 10, 12, 16, 20],
    # 'lamb': np.arange(4, 12),
}

key = 'lamb'
val_list = np.arange(4, 12)

for condition in dict_product(hyper_dict):

    key_psnr_list = list()
    for val in val_list:
        key_condition = {key: val}
        for im_name in os.listdir(search_dir):
            if is_contain_all(im_name, im_na, condition, key_condition):
                kv = decode_res(im_name)
                key_psnr_list.append(kv['psnr'])
    print(condition)
    print(key_psnr_list)

    l1 = plt.plot(val_list, key_psnr_list, 'g--', label='psnr')

    for x, psnr in zip(val_list, key_psnr_list):
        plt.text(x, psnr, 'cpp' + '%.3f' % psnr, ha='center', va='top', fontsize=10, color='g')
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.title(str(condition))
    plt.xlabel(key)
    plt.ylabel('psnr')
    plt.legend()
    os.makedirs(key, exist_ok=True)
    plt.savefig(key + '/' + dict2name(condition) + 'maxPsnr_' + str(max(key_psnr_list)) + '.png')
    # plt.show()
    plt.close()
