# From https://www.geeksforgeeks.org/feature-detection-and-matching-with-opencv-python/
# Also used ChatGPT for fetching list of features in the image.

import cv2
import numpy as np

# This function extracts features from the image.
# Returns new image that points features, as well as a list of where all the features are located in image.
def selectFeaturesFromImage(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image) 

    dst = cv2.cornerHarris(gray_image, blockSize=2, ksize=5, k=0.01) 
    
    feature_coords = np.argwhere(dst > 0.01 * dst.max())


    dst = cv2.dilate(dst, None) 
    #image[dst > 0.01 * dst.max()] = [0, 255, 0] 
    return image, feature_coords