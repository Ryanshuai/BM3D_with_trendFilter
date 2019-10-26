import os
import cv2
import math
import prox_tv as ptv

from utils import add_gaussian_noise

here = os.path.dirname(os.path.abspath(__file__))
im = cv2.imread(here + '/colors.png', cv2.IMREAD_GRAYSCALE)

sigma = 30
noisy_im = add_gaussian_noise(im, sigma)
im_h, im_w = noisy_im.shape

lam = (sigma / 255) * math.sqrt(math.log(im_h * im_w))
print(lam)
res_im = ptv.tv1_2d(noisy_im, lam)

cv2.imwrite('res_im.png', res_im)
