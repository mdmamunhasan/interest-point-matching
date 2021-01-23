import os
import cv2
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import skimage
import pywt


def get_magnitude(image):
    dft = np.fft.fft2(image)
    fshift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    # dft_shift = np.fft.fftshift(dft)
    # magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    return magnitude_spectrum


def get_patch_image(image, patch_center):
    # define some values
    patch_scale = 0.23

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


def draw_circle(img, corners):
    for i in range(1, len(corners)):
        print(corners[i, 0])
        cv2.circle(img, (int(corners[i, 0]), int(corners[i, 1])), 7, (0, 255, 0), 2)


def get_key_points(cords, point_size=5):
    # convert coordinates to Keypoint type
    kp = [cv2.KeyPoint(crd[0], crd[1], point_size) for crd in cords]
    return kp


def get_keypoints_with_descriptor(img, fast=None, star=None, orb=None, sift=None, surf=None):
    if fast:
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(img, None)
    elif star:
        star = cv2.xfeatures2d.StarDetector_create()
        kp = star.detect(img, None)
    else:
        corners, dst, gray = get_harris_corners(img)
        kp = get_key_points(corners, 5)

    if orb:
        orb = cv2.ORB_create(100, patchSize=30)
        kp, des = orb.detectAndCompute(img, None)
    elif sift:
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
    elif surf:
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, nOctaves=10)
        kp, des = surf.detectAndCompute(img, None)
    else:
        corners, dst, gray = get_harris_corners(img)
        kp = get_key_points(corners, 5)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp, des = brief.compute(img, kp)

    return kp, des


def find_homography(matches, kp1, kp2):
    points1 = np.zeros([len(matches), 2], dtype=np.float32)
    points2 = np.zeros([len(matches), 2], dtype=np.float32)
    for i, m in enumerate(matches):
        try:
            points1[i, :] = kp1[m.queryIdx].pt
            points2[i, :] = kp2[m.queryIdx].pt
        except IndexError as error:
            continue

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    # h, mask = cv2.findHomography(points1, points2)

    return h


def main():
    img1 = cv2.imread(os.path.join("images", "cv_cover1.jpg"))
    img2 = cv2.imread(os.path.join("images", "cv_desk.png"))
    # rows, cols = img1.shape[:2]
    # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    # img2 = cv2.warpAffine(img1, M, (cols, rows))

    kp1, des1 = get_keypoints_with_descriptor(img1)
    kp2, des2 = get_keypoints_with_descriptor(img2)

    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_SL2)
    matches = matcher.match(des1, des2, None)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)

    # M = find_homography(matches, kp2, kp1)

    points1 = np.array([
        [0, 0], [294, 0], [294, 400], [0, 400]
    ])
    points2 = np.array([
        [242, 194], [496, 189], [582, 484], [154, 488]
    ])
    M, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    height, width, channel = img1.shape
    img4 = cv2.warpPerspective(img2, M, (width, height))

    print(M)

    # img1[dst1 > 0.1 * dst1.max()] = [0, 0, 255]
    # img2[dst2 > 0.1 * dst2.max()] = [0, 0, 255]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()
    ax[0].imshow(img1)
    ax[0].set_title("Original")
    ax[1].imshow(img2)
    ax[1].set_title("Template")
    fig.tight_layout()
    plt.show()

    ax = plt.subplot()
    ax.imshow(img3)
    ax.set_title("Matched")
    plt.show()

    ax = plt.subplot()
    ax.imshow(img4)
    ax.set_title("Recovered")
    plt.show()


if __name__ == '__main__':
    main()
