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
        save_im_name += 'psnr' + str(psnr)[:8] + '.png'
        save_dir = 'all_3d_tf_res'
        im_path = os.path.join(save_dir, save_im_name)
        cv2.imwrite(im_path, im_denoised)
        print(im_path, '\tsaved')


if __name__ == '__main__':
    # im_path, sigma,
    # key_parameter
    # n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, lamb

    fix_hyper = {
        'im_dir': 'noisy_image_and_1st_res',
        'nH': 16,
        'kH': 8,
        'NH': 16,
        'pH': 3,
        'tauMatchH': 2500,
        'useSDH': False,
        'tau_2DH': 'BIOR',
        'lambda3DH': 2.7,

        # 'nW': 16,
        # 'kW': 8,
        # 'NW': 8,
        'pW': 3,
        'tauMatchW': 400,
        'useSDW': True,
    }

    float_hyper = {
        'im_name': ['Cameraman.png'],
        # 'sigma': [2, 5, 10, 20, 30, 40, 60, 80, 100],
        'sigma': [20],
        'nW': [32],
        'kW': [16],
        'NW': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],

        'lamb': (np.array([1, 2, 3, 4, 6, 8])[np.newaxis, :] * np.power(10., np.arange(0, 2))[:,
                                                               np.newaxis]).flatten(),
        # 'lamb': (np.array([1, 2, 3, 4, 6, 8])[np.newaxis, :] * np.power(10., np.arange(0, 2))[:, np.newaxis]).flatten(),
        # 'lamb': np.power(10., np.arange(0, 4)),
        # 'lamb': np.array([6, 6.5, 7, 7.5, 8, 8.5, 9]),
        # 'lamb': np.arange(8.0, 8.2, 0.02),
    }

    hyper = Hyper(fix_hyper, float_hyper)
    hyper.whole_round()
