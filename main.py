import os
import cv2
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

img1 = cv2.imread(os.path.join("images", "cv_desk.png"))
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray1 = np.float32(gray1)
# dst1 = cv2.cornerHarris(gray1, 2, 3, 0.2)
# result is dilated for marking the corners, not important
# dst1 = cv2.dilate(dst1, None)

# Threshold for an optimal value, it may vary depending on the image.
# img1[dst1 > 0.01 * dst1.max()] = [0, 0, 255]

img2 = cv2.imread(os.path.join("images", "cv_cover1.jpg"))
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# gray2 = np.float32(gray2)
# dst2 = cv2.cornerHarris(gray2, 2, 3, 0.2)
# result is dilated for marking the corners, not important
# dst2 = cv2.dilate(dst2, None)

# Threshold for an optimal value, it may vary depending on the image.
# img2[dst2 > 0.01 * dst2.max()] = [0, 0, 255]

orb = cv2.ORB_create(50)
kp1, ds1 = orb.detectAndCompute(img1, None)
kp2, ds2 = orb.detectAndCompute(img2, None)

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

matches = matcher.match(ds1, ds2, None)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)

points1 = np.zeros([len(matches), 2], dtype=np.float32)
points2 = np.zeros([len(matches), 2], dtype=np.float32)
for i, m in enumerate(matches):
    points1[i, :] = kp1[m.queryIdx].pt
    points2[i, :] = kp2[m.queryIdx].pt

h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

height, width, channel = img1.shape
img4 = cv2.warpPerspective(img2, h, (width, height))

# cv2.imshow('dst1', img1)
# cv2.imshow('dst2', img2)
cv2.imshow('img3', img3)
cv2.imshow('img4', img4)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
