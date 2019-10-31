import itertools
import os
import cv2
import numpy as np
from bm3d import hyper_run_bm3d_tf


class Hyper:
    def __init__(self, fix_hyper: dict, float_hyper: dict):
        self.fix_hyper = fix_hyper
        self.hpyer_keys = float_hyper.keys()
        self.product = itertools.product(*float_hyper.values())

    def whole_round(self):
        for element in self.product:
            product_parameter = dict(zip(self.hpyer_keys, element))
            self.process_function(fix_hyper, product_parameter)

    def process_function(self, fix_hyper, product_parameter):
        save_im_name = ''
        for i, (k, v) in enumerate(product_parameter.items()):
            if k == 'im_name':
                save_im_name += v[:-4] + '_'
            else:
                save_im_name += k + str(v) + '_'

        _, im_denoised, psnr = hyper_run_bm3d_tf(**fix_hyper, **product_parameter)
        save_im_name += 'psnr' + str(psnr)
        save_dir = 'all_3d_tf_res'
        cv2.imwrite(os.path.join(save_dir, save_im_name), im_denoised)


if __name__ == '__main__':
    # im_path, sigma,
    # key_parameter
    # n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, lamb

    fix_hyper = {
        'im_dir': 'test_data/image',
        'n_H': 16,
        'k_H': 8,
        'N_H': 16,
        'p_H': 3,
        'tauMatch_H': 2500,
        'useSD_H': False,
        'tau_2D_H': 'BIOR',
        'lambda3D_H': 2.7,

        'n_W': 16,
        'k_W': 8,
        'N_W': 8,
        'p_W': 3,
        'tauMatch_W': 400,
        'useSD_W': True,
    }

    float_hyper = {
        'im_name': ['Cameraman.png'],
        'sigma': [2, 5, 10, 20, 30, 40, 60, 80, 100],

        'lamb': (np.array([1, 2, 3, 4, 6, 8])[np.newaxis, :] * np.power(10., np.arange(-9, 0))[:, np.newaxis]).flatten()
    }

    hyper = Hyper(fix_hyper, float_hyper)
    hyper.whole_round()
