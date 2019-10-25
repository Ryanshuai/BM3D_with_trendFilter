import numpy as np
import cvxpy as cp


def regularizer(beta):
    rows, cols = beta.shape
    sum_across_rows = cp.tv(beta[:, 0])
    for i in range(1, cols):
        sum_across_rows = sum_across_rows + cp.tv(beta[:, i])
    sum_across_cols = cp.tv(beta[0, :])
    for i in range(1, rows):
        sum_across_cols = sum_across_cols + cp.tv(beta[i, :])
    return sum_across_rows + sum_across_cols


def objective_fn(X, beta, lambda_val):
    return 0.5 * cp.norm(X - beta, "fro") ** 2 + lambda_val * regularizer(beta)


def trend_filter(noisy_image, lambda_val):
    estimate_image = cp.Variable(noisy_image.shape)
    lambd = cp.Parameter(nonneg=True)
    lambd.value = lambda_val
    problem = cp.Problem(cp.Minimize(objective_fn(noisy_image, estimate_image, lambd)))
    f = problem.solve()
    print(f)
    res = estimate_image.value
    return res  # plt.imsave('toy_out.png',out)


if __name__ == '__main__':
    import cv2
    from utils import add_gaussian_noise

    im = cv2.imread('k0_im.png', cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (249, 249))
    noisy_im = add_gaussian_noise(im, sigma=30)

    cv2.imshow('noisy_im', noisy_im)

    res = trend_filter(noisy_im, 0)

    res_im = res.astype(np.uint8)
    cv2.imshow('res_im', res_im)
    cv2.waitKey()

