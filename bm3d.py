from utils import add_gaussian_noise, symetrize
from psnr import compute_psnr
from bm3d_1st_step import bm3d_1st_step
from bm3d_2nd_step import bm3d_2nd_step
from bm3d_2nd_step_version_2 import bm3d_2nd_step as bm3d_2nd_step_tf
save_dir = 'tf_res_version_2'


def run_bm3d(noisy_im, sigma,
             n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
             n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W):
    k_H = 8 if (tau_2D_H == 'BIOR' or sigma < 40.) else 12
    k_W = 8 if (tau_2D_W == 'BIOR' or sigma < 40.) else 12

    noisy_im_p = symetrize(noisy_im, n_H)
    img_basic = bm3d_1st_step(sigma, noisy_im_p, n_H, k_H, N_H, p_H, lambda3D_H, tauMatch_H, useSD_H, tau_2D_H)
    img_basic = img_basic[n_H: -n_H, n_H: -n_H]

    img_basic_p = symetrize(img_basic, n_W)
    noisy_im_p = symetrize(noisy_im, n_W)
    img_denoised = bm3d_2nd_step(sigma, noisy_im_p, img_basic_p, n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)
    img_denoised = img_denoised[n_W: -n_W, n_W: -n_W]

    return img_basic, img_denoised


def run_bm3d_tf(noisy_im, sigma,
             n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
             n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W, lam):
    k_H = 8 if (tau_2D_H == 'BIOR' or sigma < 40.) else 12
    k_W = 8 if (tau_2D_W == 'BIOR' or sigma < 40.) else 12

    noisy_im_p = symetrize(noisy_im, n_H)
    img_basic = bm3d_1st_step(sigma, noisy_im_p, n_H, k_H, N_H, p_H, lambda3D_H, tauMatch_H, useSD_H, tau_2D_H)
    img_basic = img_basic[n_H: -n_H, n_H: -n_H]

    img_basic_p = symetrize(img_basic, n_W)
    noisy_im_p = symetrize(noisy_im, n_W)
    img_denoised = bm3d_2nd_step_tf(sigma, noisy_im_p, img_basic_p, n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W, lam)
    img_denoised = img_denoised[n_W: -n_W, n_W: -n_W]

    return img_basic, img_denoised


if __name__ == '__main__':
    import os
    import cv2
    import numpy as np

    sigma = 20

    # <hyper parameter> -------------------------------------------------------------------------------
    n_H = 16
    k_H = 8
    N_H = 16
    p_H = 3
    lambda3D_H = 2.7  # ! Threshold for Hard Thresholding
    tauMatch_H = 2500 if sigma < 35 else 5000  # ! threshold determinates similarity between patches
    useSD_H = False
    tau_2D_H = 'BIOR'

    n_W = 16
    k_W = 8
    N_W = 32
    p_W = 3
    tauMatch_W = 400 if sigma < 35 else 3500  # ! threshold determinates similarity between patches
    useSD_W = True
    tau_2D_W = 'DCT'
    # <\ hyper parameter> -----------------------------------------------------------------------------

    im_path = 'test_data/image/Cameraman.png'
    im_name = im_path.split('/')[-1]
    os.makedirs(save_dir, exist_ok=True)

    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    noisy_im = add_gaussian_noise(im, sigma, seed=1)

    im1, im2 = run_bm3d(noisy_im, sigma,
                        n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
                        n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)

    psnr_1st = compute_psnr(im, im1)
    psnr_2nd = compute_psnr(im, im2)

    im1 = np.clip(im1, 0, 255)
    im2 = np.clip(im2, 0, 255)
    im1 = im1.astype(np.uint8)
    im2 = im2.astype(np.uint8)

    save_name = im_name[:-4] + '_s' + str(sigma) + '_1_P' + str(round(psnr_1st, 3)) + '.png'
    cv2.imwrite(os.path.join(save_dir, save_name), im1)
    save_name = im_name[:-4] + '_s' + str(sigma) + '_2_P' + str(round(psnr_2nd, 3)) + '.png'
    cv2.imwrite(os.path.join(save_dir, save_name), im2)

    # <hyper parameter> -------------------------------------------------------------------------------
    tau_2D_W = ''
    # <\ hyper parameter> -----------------------------------------------------------------------------

    # for lam in [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50]:
    for lam in [1, 2, 3, 5, 10, 20, 30, 50]:
        print('lam: ', lam)
        im1, im2 = run_bm3d_tf(noisy_im, sigma,
                            n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
                            n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W, lam)

        psnr_2nd_tf = compute_psnr(im, im2)

        im1 = np.clip(im1, 0, 255)
        im2 = np.clip(im2, 0, 255)
        im1 = im1.astype(np.uint8)
        im2 = im2.astype(np.uint8)

        save_name = im_name[:-4] + '_s' + str(sigma) + '_2_lam' + str(lam) + '_P' + str(round(psnr_2nd_tf, 3)) + '.png'
        cv2.imwrite(os.path.join(save_dir, save_name), im2)
