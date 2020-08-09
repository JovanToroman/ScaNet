import os
import sqlite3 as sql

import cv2 as cv
import numpy as np

from db import insert_into_db

# this is stage 1 in the processing pipeline

pts_dst_dict = {0: np.array([[0, 100], [0, 0], [100, 0], [100, 100]]),
                1: np.array([[860, 100], [860, 0], [960, 0], [960, 100]]),
                2: np.array([[0, 740], [0, 640], [100, 640], [100, 740]]),
                3: np.array([[430, 740], [430, 640], [530, 640], [530, 740]]),
                4: np.array([[860, 740], [860, 640], [960, 640], [960, 740]]),
                5: np.array([[430, 100], [430, 0], [530, 0], [530, 100]]),
                6: np.array([[860, 420], [860, 320], [960, 320], [960, 420]]),
                7: np.array([[0, 420], [0, 320], [100, 320], [100, 420]])}

# how many times to multiply dst point value to scale image properly
scale_factor = 3
offset = 200


# noinspection PyUnresolvedReferences
def transform(src_image_path, dst_image_path, output_image_path = None):
    img_test = cv.imread(src_image_path)
    # img_orig = cv.imread(dst_image_path)

    if os.path.exists(output_image_path) or (np.array(img_test).ndim and np.array(img_test).size) == 0: # if we already processed an image or if it was not received properly and is empty
        return
    #Load the dictionary that was used to generate the markers.
    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_250)

    # Initialize the detector parameters using default values
    parameters =  cv.aruco.DetectorParameters_create()

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(img_test, dictionary, parameters=parameters)
    # marker_ids_list = list(markerIds)

    # cv.imshow("asdasdsadad",cv.resize(cv.drawMarker(img_orig,
    #                                                 (int(img_orig.shape[1]/2 - 100), img_orig.shape[0]), thickness=50, color=1), (800,600)))
    #
    # cv.waitKey()

    # cv.imshow("asdasdsadad",cv.resize(cv.drawContours(img_test,
    #                           np.array([markerCorners[0][0], markerCorners[1][0], markerCorners[2][0], markerCorners[3][0]])
    #                           .astype('int'), color=5, contourIdx=-1, thickness=5), (1024,768)))
    #
    # cv.waitKey()

    if (np.array(markerIds).ndim and np.array(markerIds).size) == 0:
        print("No aruco markers detected for image {}".format(src_image_path))
        return (1, None)

    # pts_src = np.array([markerCorners[2][0][2], markerCorners[0][0][3], markerCorners[1][0][0], markerCorners[3][0][1]]).astype('int')
    pts_src = np.array(markerCorners[0]).astype('int')

    # create a dict containing destination points for different markers
    markerId = None
    for marker_id in markerIds:
        if marker_id[0] < 8:
            markerId = marker_id[0]
            break

    if markerId == None:
        print("No valid aruco markers detected for image {}".format(src_image_path))
        return (2, None)

    print(markerId)

    pts_dst = (pts_dst_dict[markerId] + offset) * 2.5

    # Calculate Homography
    h, status = cv.findHomography(pts_src, pts_dst)


    # Warp source image to destination based on homography
    warped_image = cv.warpPerspective(img_test, h, ((960 + offset) * scale_factor, (740 + offset) * scale_factor)) # 2 x offset because we have it from both sides od the axes
    # warped_image = warped_image[50:warped_image.shape[0] - 50,50:warped_image.shape[1] - 50] # crop some of the outer whitespace
    # gray = cv.cvtColor(warped_image,cv.COLOR_BGR2GRAY)
    # _,thresh = cv.threshold(gray,100,255,cv.THRESH_BINARY)
    # thresh = 255*(thresh < 128).astype(np.uint8)
    # coords = cv.findNonZero(thresh)

    # bi = cv.bilateralFilter(gray, 5, 75, 75)
    # dst = cv.cornerHarris(bi, 2, 3, 0.04)
    # dst = cv.dilate(dst,None)
    #
    # warped_image[dst > 0.01 * dst.max()] = [0, 0, 255]   #--- [0, 0, 255] --> Red ---
    # cv.imshow('dst', warped_image)
    # cv.waitKey()

    # contours,hierarchy = cv.findContours(gray,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    # cnt = contours[0]
    # x,y,w,h = cv.boundingRect(coords)
    # crop = warped_image[y:y+h,x:x+w]

    # cv.imshow("Source Image", img_test)
    # cv.imshow("Destination Image", img_orig)
    # cv.imshow("Warped Source Image", cv.resize(warped_image, (800,600)))

    if output_image_path:
        cv.imwrite(output_image_path, warped_image)
        return (0, None)

    # cv.waitKey()


phones = ['HTC', 'LG_G3', 'Motorola', 'Samsung_A3', 'Samsung_S20', 'Sony_Xperia']

conn = sql.connect('../image.sqlite')

for phone in phones:
    path_src = os.path.join('../captured_images', phone)
    path_dst = 'origim'
    path_out = os.path.join('../captured_images/transposed', phone)

    for image in os.listdir(path_src):
        print(image)
        original_image_name = image.split('_')[0] + '.jpg'
        result = transform(os.path.join(path_src, image), os.path.join(path_dst, original_image_name),
                  os.path.join(path_out, image))
        if result:
            status, ssim = result
            insert_into_db(image, status, 1, None, None, phone, conn)