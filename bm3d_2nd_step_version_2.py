import numpy as np
import cv2
import prox_tv as ptv

from utils import ind_initialize, get_kaiserWindow, sd_weighting
from precompute_BM import precompute_BM
from bior_2d import bior_2d_forward, bior_2d_reverse
from dct_2d import dct_2d_forward, dct_2d_reverse
from image_to_patches import image2patches
from build_3D_group import build_3D_group
from wiener_filtering_hadamard import wiener_filtering_hadamard


def bm3d_2nd_step(sigma, img_noisy, img_basic, nWien, kWien, NWien, pWien, tauMatch, useSD, tau_2D, lam):
    height, width = img_noisy.shape[0], img_noisy.shape[1]

    row_ind = ind_initialize(height - kWien + 1, nWien, pWien)
    column_ind = ind_initialize(width - kWien + 1, nWien, pWien)

    kaiserWindow = get_kaiserWindow(kWien)
    ri_rj_N__ni_nj, threshold_count = precompute_BM(img_basic, kHW=kWien, NHW=NWien, nHW=nWien, tauMatch=tauMatch)
    group_len = int(np.sum(threshold_count))
    group_3D_table = np.zeros((group_len, kWien, kWien))
    weight_table = np.ones((height, width))

    noisy_patches = image2patches(img_noisy, k=kWien, p=pWien)  # i_j_ipatch_jpatch__v
    basic_patches = image2patches(img_basic, k=kWien, p=pWien)  # i_j_ipatch_jpatch__v
    if tau_2D == 'DCT':
        fre_noisy_patches = dct_2d_forward(noisy_patches)
        fre_basic_patches = dct_2d_forward(basic_patches)
    elif tau_2D == 'BIOR':  # 'BIOR'
        fre_noisy_patches = bior_2d_forward(noisy_patches)
        fre_basic_patches = bior_2d_forward(basic_patches)
    else:
        fre_noisy_patches = np.zeros_like(noisy_patches)
        for i, patch in enumerate(noisy_patches):
            fre_noisy_patches[i] = ptv.tv1_2d(patch, lam)
        fre_basic_patches = np.zeros_like(basic_patches)
        for i, patch in enumerate(basic_patches):
            fre_basic_patches[i] = ptv.tv1_2d(patch, lam)

    fre_basic_patches = fre_basic_patches.reshape((height - kWien + 1, height - kWien + 1, kWien, kWien))

    acc_pointer = 0
    for i_r in row_ind:
        for j_r in column_ind:
            nSx_r = threshold_count[i_r, j_r]
            group_3D_est = build_3D_group(fre_basic_patches, ri_rj_N__ni_nj[i_r, j_r], nSx_r)
            group_3D = group_3D_est
            group_3D = group_3D.transpose((2, 0, 1))

            group_3D_table[acc_pointer:acc_pointer + nSx_r] = group_3D
            acc_pointer += nSx_r

            if useSD:
                weight = sd_weighting(group_3D)

            weight_table[i_r, j_r] = weight

    if tau_2D == 'DCT':
        group_3D_table = dct_2d_reverse(group_3D_table)
    elif tau_2D == 'BIOR':  # 'BIOR'
        group_3D_table = bior_2d_reverse(group_3D_table)
    else:
        pass

    # for i in range(1000):
    #     patch = group_3D_table[i]
    #     print(i, '----------------------------')
    #     print(patch)
    #     cv2.imshow('', patch.astype(np.uint8))
    #     cv2.waitKey()

    group_3D_table *= kaiserWindow

    numerator = np.zeros_like(img_noisy, dtype=np.float64)
    denominator = np.zeros_like(img_noisy, dtype=np.float64)
    acc_pointer = 0
    for i_r in row_ind:
        for j_r in column_ind:
            if i_r == 264 and j_r == 264:
                print()

            nSx_r = threshold_count[i_r, j_r]
            N_ni_nj = ri_rj_N__ni_nj[i_r, j_r]
            group_3D = group_3D_table[acc_pointer:acc_pointer + nSx_r]
            acc_pointer += nSx_r
            weight = weight_table[i_r, j_r]
            for n in range(nSx_r):
                ni, nj = N_ni_nj[n]
                patch = group_3D[n]

                numerator[ni:ni + kWien, nj:nj + kWien] += patch * weight
                denominator[ni:ni + kWien, nj:nj + kWien] += kaiserWindow * weight

    img_denoised = numerator / denominator
    return img_denoised


if __name__ == '__main__':
    from psnr import compute_psnr
    from utils import add_gaussian_noise, symetrize

    # <hyper parameter> -------------------------------------------------------------------------------
    sigma = 20

    nWien = 16
    kWien = 8
    NWien = 16
    pWien = 3
    tauMatchWien = 400 if sigma < 35 else 3500  # ! threshold determinates similarity between patches
    useSD_w = True
    tau_2D_wien = 'DCT'
    # <\ hyper parameter> -----------------------------------------------------------------------------

    img = cv2.imread('Cameraman256.png', cv2.IMREAD_GRAYSCALE)
    img_noisy = cv2.imread('image_noise.png', cv2.IMREAD_GRAYSCALE)
    img_basic = cv2.imread('y_basic.png', cv2.IMREAD_GRAYSCALE)

    img_basic_p = symetrize(img_basic, nWien)
    img_noisy_p = symetrize(img_noisy, nWien)
    img_denoised = bm3d_2nd_step(sigma, img_noisy_p, img_basic_p, nWien, kWien, NWien, pWien, tauMatchWien, useSD_w,
                                 tau_2D_wien)
    img_denoised = img_denoised[nWien: -nWien, nWien: -nWien]

    psnr_2st = compute_psnr(img, img_denoised)
    print('img and img_denoised PSNR: ', psnr_2st)
    # cv2.imwrite('y_final.png', img_denoised.astype(np.uint8))
