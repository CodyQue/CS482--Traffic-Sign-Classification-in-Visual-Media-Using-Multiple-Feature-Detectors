# https://www.youtube.com/watch?v=uEJ71VlUmMQ&ab_channel=Computerphile
# Also used ChatGPT to help compute the integral image

import cv2
import numpy as np

# This function computes the integral image
def calculateIntegralImage(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print('Gray image: ', gray_image)
    gray_image = np.array(gray_image)
    cumsum_rows = np.cumsum(gray_image, axis=0)
    cumsum_cols = np.cumsum(cumsum_rows, axis=1)  
    #print('Cum Sum image: ', cumsum_cols)
    return cumsum_cols

# This function loops through all the features of the image and finds the Haar Features
# For this project, only the rectangular Haar Featurs will be detected instead of tilted.
# https://en.wikipedia.org/wiki/Haar-like_feature
# https://www.youtube.com/watch?v=ZSqg-fZJ9tQ&ab_channel=FirstPrinciplesofComputerVision

def computeHaarFeatures(integralImage, features):
    print('Length: ', features.shape)
    rowCount = integralImage.shape[0]
    first_half = integralImage[:rowCount//2]
    second_half = integralImage[rowCount//2:]
    print('First Half: ', first_half)
    print('Second Half: ', second_half)
    print(second_half[-1, -1] - first_half[-1, -1])
    for i in features:
        #print(i)
        x = i[0]
        y = i[1]
        #print(integralImage[x][y])