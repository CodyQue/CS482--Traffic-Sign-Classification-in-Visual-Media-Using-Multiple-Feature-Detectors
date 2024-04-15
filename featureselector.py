# From https://www.geeksforgeeks.org/feature-detection-and-matching-with-opencv-python/

import cv2
import numpy as np

# Load the image
image = cv2.imread("stopsign.jpg") 
image = cv2.resize(image, (500, 500))
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = np.float32(gray_image) 

dst = cv2.cornerHarris(gray_image, blockSize=3, ksize=5, k=0.02) 

dst = cv2.dilate(dst, None) 
#image[dst > 0.01 * dst.max()] = [0, 255, 0] 

#cv2.imshow('haris_corner', image) 
cv2.imshow('haris_corner', dst) 
cv2.waitKey() 