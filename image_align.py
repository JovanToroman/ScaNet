from __future__ import print_function
import cv2
import numpy as np
import tensorflow as tf

import os

from utils import calc_mse

# this is stage 2 in the processing pipeline
# status 0 - success; status 1 - no alignment; status 2 - bad alignment (insufficient ssim)

MAX_FEATURES = 10000
GOOD_MATCH_PERCENT = 0.05


def alignImages(im1, im2):
    try:
        # Convert images to grayscale
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
        # cv2.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width, channels = im2.shape
        im1Reg = cv2.warpPerspective(im1, h, (width, height))
    except Exception as e:
        return [], []

    return im1Reg, h


def align_image(original_path, scanned_path, output_path):
    if os.path.exists(output_path):
        return

    # Read reference image
    print("Reading reference image : ", original_path)
    imReference = cv2.imread(original_path, cv2.IMREAD_COLOR)

    # Read image to be aligned
    # print("Reading image to align : ", scanned_path);
    im = cv2.imread(scanned_path, cv2.IMREAD_COLOR)

    # print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    if len(imReg) == 0:
        print("Failed alignment for {}".format(scanned_path))
        return (1, None)

    ssim = float(tf.image.ssim(tf.convert_to_tensor(imReference), tf.convert_to_tensor(imReg), 1.0))
    print('Ssim for image {} is {}'.format(output_path, ssim))
    mse = calc_mse(imReference, imReg)

    # Write aligned image to disk.
    # print("Saving aligned image : ", output_path);
    if ssim > 0.1: # if ssim < some threshold then image was not properly aligned
        cv2.imwrite(output_path.split('.')[0] + '_' + str(mse) + '_' + str(ssim) + '_' + '.' + output_path.split('.')[1], imReg)
        return (0, ssim, mse)
    else:
        print('Bad alignment of image {}'.format(output_path))
        return (2, ssim)

    # Print estimated homography
    # print("Estimated homography : \n", h)
