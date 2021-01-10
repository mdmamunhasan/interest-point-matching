import os
import cv2
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


def get_keypoints_with_descriptor(img, fast=None, orb=None, sift=None, surf=None):
    if fast:
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(img, None)
    else:
        star = cv2.xfeatures2d.StarDetector_create()
        kp = star.detect(img, None)

    if orb:
        orb = cv2.ORB_create(10)
        kp, des = orb.detectAndCompute(img, None)
    elif sift:
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
    elif surf:
        surf = cv2.xfeatures2d.SURF_create()
        kp, des = surf.detectAndCompute(img, None)
    else:
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp, des = brief.compute(img, kp)

    return kp, des


def find_homography(matches, kp1, kp2):
    points1 = np.zeros([len(matches), 2], dtype=np.float32)
    points2 = np.zeros([len(matches), 2], dtype=np.float32)
    for i, m in enumerate(matches):
        points1[i, :] = kp1[m.queryIdx].pt
        points2[i, :] = kp2[m.queryIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    return h


def main():
    img1 = cv2.imread(os.path.join("images", "cv_desk.png"))
    corners1, dst1, gray1 = get_harris_corners(img1)

    img2 = cv2.imread(os.path.join("images", "cv_cover1.jpg"))
    corners2, dst2, gray2 = get_harris_corners(img2)

    kp1, des1 = get_keypoints_with_descriptor(img1)
    kp2, des2 = get_keypoints_with_descriptor(img2)

    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_SL2)
    matches = matcher.match(des1, des2, None)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)

    h = find_homography(matches, kp1, kp2)
    height, width, channel = img1.shape
    img4 = cv2.warpPerspective(img2, h, (width, height))

    img1[dst1 > 0.1 * dst1.max()] = [0, 0, 255]
    img2[dst2 > 0.1 * dst2.max()] = [0, 0, 255]

    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    cv2.imshow('img3', img3)
    # cv2.imshow('img4', img4)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
