import os
import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy
from skimage.color import rgb2gray
from skimage import data
from skimage import filters
from skimage import io
from skimage import img_as_float
from skimage import feature
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse
from skimage.feature import CENSURE

# Input Image 1
filename1 = os.path.join("images", 'IMG_0320.JPG')
img1 = io.imread(filename1)
img1 = img_as_float(img1)
gray1 = rgb2gray(img1)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(img1)
ax[0].set_title("Original")
ax[1].imshow(gray1, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")
fig.tight_layout()
plt.show()

# Input Image 2
filename2 = os.path.join("images", 'IMG_0321.JPG')
img2 = io.imread(filename2)
img2 = img_as_float(img2)
gray2 = rgb2gray(img2)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(img2)
ax[0].set_title("Original")
ax[1].imshow(gray2, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")
fig.tight_layout()
plt.show()

# Harris Corner
coords1 = corner_peaks(corner_harris(gray1), min_distance=5, threshold_rel=0.02)
coords_subpix1 = corner_subpix(gray1, coords1, window_size=10)

coords2 = corner_peaks(corner_harris(gray2), min_distance=5, threshold_rel=0.02)
coords_subpix2 = corner_subpix(gray2, coords2, window_size=10)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(gray1, cmap=plt.cm.gray)
ax[0].plot(coords1[:, 1], coords1[:, 0], color='cyan', marker='o', linestyle='None', markersize=6)
ax[0].plot(coords_subpix1[:, 1], coords_subpix1[:, 0], '+r', markersize=15)
ax[0].set_title("Original")
ax[1].imshow(gray2, cmap=plt.cm.gray)
ax[1].plot(coords2[:, 1], coords2[:, 0], color='cyan', marker='o', linestyle='None', markersize=6)
ax[1].plot(coords_subpix2[:, 1], coords_subpix2[:, 0], '+r', markersize=15)
ax[1].set_title("Template")
fig.tight_layout()
plt.show()

# Censure Keypoints

censure = CENSURE()
censure.detect(gray1)
kp1 = censure.keypoints
censure.detect(gray2)
kp2 = censure.keypoints

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(gray1, cmap=plt.cm.gray)
ax[0].plot(kp1[:, 1], kp1[:, 0], color='cyan', marker='o', linestyle='None', markersize=6)
ax[0].set_title("Original")
ax[1].imshow(gray2, cmap=plt.cm.gray)
ax[1].plot(kp2[:, 1], kp2[:, 0], color='cyan', marker='o', linestyle='None', markersize=6)
ax[1].set_title("Template")
fig.tight_layout()
plt.show()
