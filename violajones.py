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
# https://www.youtube.com/watch?v=p9vq90NYHMs&ab_channel=YoussefShakrina

def computeHaarFeatures(integralImage, features, image):
    #print('Length: ', features.shape)
    rowCount = integralImage.shape[0]
    first_half = integralImage[:rowCount//2]
    second_half = integralImage[rowCount//2:]
    #print('First Half: ', first_half)
    #print('Second Half: ', second_half)
    #print(second_half[-1, -1] - first_half[-1, -1])
    for i in features:
        #print(i)
        x = i[0]
        y = i[1]
        for j in range(5):
            h = 20 + (5 * j)
            sub1 = integralImage[x-(h+1)][y+h]
            sub2 = integralImage[x+h][y-(h+1)]
            add1 = integralImage[x-(h+1)][y-(h+1)]
            arr = integralImage[x-h:x+h,y-h:y+h]
            total = arr[-1][-1]
            print(arr)
            print(arr.shape)
            print(total)
            #print('X: ', x, ', Y: ', y)
            print('Sub 1: ', sub1, 'Sub 2', sub2, 'Add 1: ', add1)
            value = int(total - sub1)
            print(value)
            print(sub2)
            value -= int(sub2)
            value += int(add1)
            print('Value: ', value)
            cv2.imshow('haris_corner', image[x-h:x+h,y-h:y+h]) 
            cv2.waitKey() 