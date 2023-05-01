#
#  ReflectionDetection.cpp
#
#  Created by Waqas Haider Sheikh on 02/06/2019.
#  Copyright © 2019 Waqas Haider Sheikh. All rights reserved.
#  Tes

import cv2 as cv
import numpy as np
import os
from sorting_contours import sort_contours as sc

'''this function detect reflection in matrix
reference: https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
 
@param source matrix
@return true/false
'''

def ReflectionDetection(source):
    gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray', gray)
    # smooth the image using a 11x11 Gaussian
    #blurred = cv.GaussianBlur(gray, (19,19), 0)
    #x = 1
    #blurred = cv.blur(gray, (x,x))
    # cv.imshow('Blurred', blurred)
    # threshold the image to reveal light regions in the blurred image
    #threshold, thresh = cv.threshold(gray, 175, 255, cv.THRESH_BINARY)
    #thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 111, 11)
    # cv.imshow('Threshold', thresh)

    # perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
    erroded = cv.erode(gray, (-1,-1), iterations=2)
    # cv.imshow('Erroded', erroded)
    dilated = cv.dilate(erroded, (-1,-1), iterations=4)
    # cv.imshow('Dilated', dilated)

    largest_radius = 0.0
    threshold = 400.0
    
    # Mencari Contour
    contours, hierarchies = cv.findContours(dilated, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    blank = np.ones(img.shape, dtype='uint8')
    cv.drawContours(blank , contours, -1, (0,0,255), 1)
    # cv.imshow('Contours Drawn', blank)
    #print(contours)
    print(f'Banyaknya kontur ialah: {len(contours)}')
    #sort contours by descending orders
    contours = sc(contours, "bottom-to-top")
    #print(enumerate(contours))
    contours = contours[0]
    #print(contours)
    #print(type(contours))
    # iterate through each contour.
    for cnt in contours:
        center, radius = cv.minEnclosingCircle(cnt)
        if radius > largest_radius:
            largest_radius = radius
    
    # release mrz thresh
    #gray.release()
    result1 = cv.inpaint(img, mask, 0.1, cv.INPAINT_TELEA)
    cv.imwrite(f"{out_path}/{out_path}_{i}", result1)
    # cv.imshow('result1', result1)
    return largest_radius > threshold

# # comparison function object
# bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
#     double i = fabs( contourArea(cv::Mat(contour1)));
#     double j = fabs( contourArea(cv::Mat(contour2)));
#     return ( i > j );
# }

folder_path = "Gmbr"
out_path = r"out1"
if not os.path.exists(out_path):
    os.mkdir(out_path)
x = 0
for i in os.listdir(folder_path):
    #i = "DSC09654.JPG"
    print(i)
    #break
    img = cv.imread(f"{folder_path}/{i}")
    # Resize gambar, beda interpolation beda hasil
    #(height, width) = img.shape[:2]
    #img = cv.resize(img, (width//10,height//10), interpolation=cv.INTER_CUBIC)
    # cv.imshow('Ori', img)

    # Define lower and uppper limits of what we call "white"
    brown_lo=np.array([210,205,205])
    brown_hi=np.array([255,255,255])

    # Mask image to only select white
    mask=cv.inRange(img,brown_lo,brown_hi)
    # cv.imshow('TES', mask)

    result = ReflectionDetection(img)
    #print(f'motor{i}, status: {result}')
    #print(f'Reflectance Status: {result}')
    # x += 1
    # if x == 1:
    #     break

cv.waitKey(0)