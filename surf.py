import os
import cv2
import pywt
import PIL
import pywt.data
import numpy as np
from skimage import feature
from skimage import transform
import matplotlib.pyplot as plt

def har_wavelet(image):
    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(image, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()


def integral_image(image):
    # cv2.cvIntegral(image)
    img = transform.integral.integral_image(image)
    plt.imshow(img)
    plt.show()


def hessian_matrix(image):
    # cv2.cvIntegral(image)
    feature.hessian_matrix(image, sigma=1, mode='constant', cval=0, order='rc')
    plt.imshow(image)
    plt.show()


def image_pyr():
    pass


def get_patch_image(image, patch_center):
    # define some values
    patch_scale = 0.15

    # calc patch position and extract the patch
    smaller_dim = np.min(image.shape[0:2])
    patch_size = int(patch_scale * smaller_dim)
    patch_x = int(patch_center[0] - patch_size / 2.)
    patch_y = int(patch_center[1] - patch_size / 2.)
    patch_image = image[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]

    return patch_image


def get_harris_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 5, 3, 0.04)
    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    return corners, dst, gray


if __name__ == '__main__':
    # Load image
    img1 = cv2.imread(os.path.join("images", "cv_cover1.jpg"))

    patches = []
    corners, _, _ = get_harris_corners(img1)
    for corner in corners:
        patch = get_patch_image(img1, corner)
        cA, (cH, cV, cD) = pywt.dwt2(patch, 'bior1.3')
        patches.append(patch)

    plt.imshow(cH)
    plt.show()
