import os
from skimage import transform
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, BRIEF, plot_matches)
from skimage.color import rgb2gray
from skimage import io
from skimage import img_as_float
import matplotlib.pyplot as plt

# Input Image
filename1 = os.path.join("images", 'cv_cover1.jpg')
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
filename2 = os.path.join("images", 'cv_desk.png')
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

tform = transform.AffineTransform(scale=(1.3, 1.1), rotation=0.5,
                                  translation=(0, -200))
gray3 = transform.warp(gray1, tform)
gray4 = transform.rotate(gray1, 180)

descriptor_extractor = BRIEF(patch_size=5)

keypoints1 = corner_peaks(corner_harris(gray1), min_distance=1, threshold_rel=0)
descriptor_extractor.extract(gray1, keypoints1)
descriptors1 = descriptor_extractor.descriptors

keypoints2 = corner_peaks(corner_harris(gray2), min_distance=1, threshold_rel=0)
descriptor_extractor.extract(gray2, keypoints2)
descriptors2 = descriptor_extractor.descriptors

keypoints3 = corner_peaks(corner_harris(gray3), min_distance=1, threshold_rel=0)
descriptor_extractor.extract(gray3, keypoints3)
descriptors3 = descriptor_extractor.descriptors

keypoints4 = corner_peaks(corner_harris(gray4), min_distance=1, threshold_rel=0)
descriptor_extractor.extract(gray4, keypoints4)
descriptors4 = descriptor_extractor.descriptors

matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
matches13 = match_descriptors(descriptors1, descriptors3, cross_check=True)
matches14 = match_descriptors(descriptors1, descriptors4, cross_check=True)
fig, ax = plt.subplots(nrows=3, ncols=1)

plt.gray()

plot_matches(ax[0], gray1, gray2, keypoints1, keypoints2, matches12[:10])
ax[0].axis('off')
ax[0].set_title("Original Image vs. Transformed Image")

plot_matches(ax[1], gray1, gray3, keypoints1, keypoints3, matches13[:10])
ax[1].axis('off')
ax[1].set_title("Original Image vs. Transformed Image")

plot_matches(ax[2], gray1, gray4, keypoints1, keypoints4, matches14[:10])
ax[2].axis('off')
ax[2].set_title("Original Image vs. Rotated Image")

plt.show()
