# https://www.youtube.com/watch?v=uEJ71VlUmMQ&ab_channel=Computerphile
# Also used ChatGPT to help compute the integral image

import cv2
import numpy as np
import pandas as pd

# This is for testing purposes
def printImageWithIndex(image):
    count = 1
    for i in image:
        print('Index: ', count, ', ', i)

# This function computes the integral image, which is the sum of all pixels from a specific point. This is necessary for computing the Haar Features since it requires sums of specific features.
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
# Returns a DataFrame containing all 4 Haar Feature and sum calculations
# https://en.wikipedia.org/wiki/Haar-like_feature
# https://www.youtube.com/watch?v=ZSqg-fZJ9tQ&ab_channel=FirstPrinciplesofComputerVision
# https://www.youtube.com/watch?v=p9vq90NYHMs&ab_channel=YoussefShakrina
def computeHaarFeatures(integralImage, features, image):
    #print('Length: ', features.shape)
    #rowCount = integralImage.shape[0]
    #print('First Half: ', first_half)
    #print('Second Half: ', second_half)
    #print(second_half[-1, -1] - first_half[-1, -1])
    df = pd.DataFrame(columns=['Aba Boost 1', 'Aba Boost 2', 'Aba Boost 3', 'Aba Boost 4'])
    for i in features:
        arr2 = []
        #print(i)
        x = i[0]
        y = i[1]
        #print('X: ', x, ', Y: ', y)
        try:
            h = 30 # Since the program will be testing the Haar features on four different sizes.
            totalPixels = totalPixelsOfFeature(integralImage, x, y, h) # This first part computes the first AdaBoost training, which consists of up-down edge detection.
            top1, bottom1 = adaBoostTraining1(integralImage, x, y, h, totalPixels)
            left1, right1 = adaBoostTraining2(integralImage, x, y, h, totalPixels)
            left, middle, right = adaBoostTraining3(integralImage, x, y, h, totalPixels)
            leftup, rightup, leftdown, rightdown = adaBoostTraining4(integralImage, x, y, h, totalPixels)
            adaBoost1Value = bottom1 - top1
            adaBoost2Value = left1-right1
            adaBoost3Value = (left + right) - middle
            adaBoost4Value = (leftup + rightdown) - (rightup + leftdown)
            #print('Aba Boost 1: ', adaBoost1Value)
            #print('Aba Boost 2: ', adaBoost2Value)
            #print('Aba Boost 3: ', adaBoost3Value)    
            #print('Aba Boost 4: ', adaBoost4Value) 
            
            #total = (adaBoost1Value + adaBoost2Value + adaBoost3Value + adaBoost4Value)
            #print('Total of all Ada Training: ', total)
            #meanAdaBoost1 += totalPixels
            #cv2.imshow('haris_corner', image[x-h:x+h,y-h:y+h]) 
            #cv2.waitKey() 
            new_row = {'Aba Boost 1': adaBoost1Value, 'Aba Boost 2': adaBoost2Value, 'Aba Boost 3': adaBoost3Value, 'Aba Boost 4': adaBoost4Value}
            
            df.loc[len(df.index)] = new_row
        except Exception as e:
            #print('Error: ', e)
            continue
        #print('Mean: ', meanAdaBoost1)
    return df
        
# This computes the total pixels around a specific feature.
# This takes in the Integral Image, the x and y coordinate of the feature, and h, how big the area should be.
def totalPixelsOfFeature(integralImage, x, y, h):
    #print('h: ', h, ', h-1: ', h+1)
    arr = integralImage[x-h:x+h,y-h:y+h] # Array up to where the feature is + h
    np.set_printoptions(threshold=np.inf)
    c = 0
    sub1 = integralImage[x+h][y-(h+1)] # Gets left total pixels.
    sub2 = integralImage[x-(h+1)][y+h] # Gets up total pixels.
    add1 = integralImage[x-(h+1)][y-(h+1)] # Gets top left total pixels, used since it double subtracts this area when subtracting sub1 and subtracting sub2. Therefore, this needs to be added back.
    total = arr[-1][-1] # Total number of pixels up to that specific feature.
    aTotal = int(total - sub1)
    aTotal -= int(sub2)
    aTotal += int(add1)
    return aTotal
    
    #arr = integralImage[x-h:x+h,y-h:y+h] # Array up to where the feature is + h
    #midpoint = arr.shape[0] // 2
    #print('Midpoint: ', midpoint)

# This computes the x-derivative (up-down) edge detection, this is used to subtract the top from the bottom, as followed for Haar Features.
# Returns the total pixels of the top half and the total pixels of the bottom half.
def adaBoostTraining1(integralImage, x, y, h, totalPixels):
    t1 = integralImage[x-1][y+h]
    t2 = integralImage[x-1][y-(h+1)]
    t3 = integralImage[x-(h+1)][y+h]
    a1 = integralImage[x-(h+1)][y-(h+1)]
    
    b1 = integralImage[x+h][y+h]
    b2 = integralImage[x+h][y-(h+1)]
    b3 = t1
    a2 = t2
    aBottom = int(b1) - int(b2) - int(b3) + int(a2)
    #print('T2: ', t2)

    # Gets total pixels of the first half
    #aTop = int(t1) - int(t2) - int(t3) + int(a1)
    
    aTop = totalPixels - aBottom

    # Gets total pixels of the second half
    #aBottom = totalPixels - aTop
    #print('Ada Test 1: Total Pixels: ', totalPixels, ', A Top: ', aTop, ', A Bottom: ', aBottom)
    return aTop, aBottom

# This computes the y-derivative edge detection, this is used to subtract the left from the right, as followed for Haar Features.
# Returns the total pixels of the top half and the total pixels of the bottom half.
def adaBoostTraining2(integralImage, x, y, h, totalPixels):
    right1 = integralImage[x+h][y+h]
    right2 = integralImage[x+h][y-1]
    right3 = integralImage[x-(h+1)][y+h]
    rightadd = integralImage[x-(h+1)][y-1]

    # Gets total pixels of the first half
    #right = int(right1) - int(right2) - int(right3) + int(rightadd)
    
    left1 = integralImage[x+h][y-1]
    left2 = integralImage[x+h][y-(h+1)]
    left3 = integralImage[x-(h+1)][y-1]
    leftadd = integralImage[x-(h+1)][y-(h+1)]
    
    left = int(left1) - int(left2) - int(left3) + int(leftadd)
    right = totalPixels - left

    #print('Ada Test 2: Total Pixels: ', totalPixels, ', A Left: ', left, ', A Right: ', right)
    return left, right

# This computes the x-axis-derivative line detection.
# For this, the area around the feature would need to be split in three different parts, the left, the middle, and the right.
# To obtain the value, the middle is subtracted by the sum of the left and the right.
def adaBoostTraining3(integralImage, x, y, h, totalPixels):

    division = h // 3 # Dividing the area around feature into three different parts, only being used for y-coordinate.
    #print('Division: ', division)
    #print('X: ', x, ', Y: ', y)
    
    middle1 = integralImage[x+h][y+division]
    #print('Middle 1 X: ', x+h, ', Y: ', y+division)
    middle2= integralImage[x+h][y-(division+1)]
    #print('Middle 2 X: ', x+h, ', Y: ', y-(division+1))
    middle3 = integralImage[x-(h+1)][y+division]
    #print('Middle 3 X: ', x-(h+1), ', Y: ', y+division)
    middleadd = integralImage[x-(h+1)][y-(division+1)]
    #print('Middle Add X: ', x-(h+1), ', Y: ', y+division)
    middle = int(middle1) - int(middle2) - int(middle3) + int(middleadd)
    #print('Middle 1: ', middle1, ', Middle 2: ', middle2, ', Middle 3: ', middle3, ', Middle Add: ', middleadd)
    
    left1 = middle2
    left2 = integralImage[x+h][y-(h+1)]
    left3 = middleadd 
    leftadd = integralImage[x-(h+1)][y-(h+1)]
    left = int(left1) - int(left2) - int(left3) + int(leftadd)
    #print('Left 1: ', left1, ', Left 2: ', left2, ', Left 3: ', left3, ', Left Add: ', leftadd)
    
    right = totalPixels - middle - left
    
    #print('Ada Test 3: Left: ', left, ', Middle: ', middle, ', Right: ', right)
    return left, middle, right


# This computes the four-rectangle feature detection for AdaBoost Training.
# For this, the area around the feature would need to be split in three different parts, the left, the middle, and the right.
# To obtain the value, the middle is subtracted by the sum of the left and the right.
def adaBoostTraining4(integralImage, x, y, h, totalPixels):
    leftup1 = integralImage[x][y-1]
    leftup2 = integralImage[x][y-(h+1)]
    leftup3 = integralImage[x-(h+1)][y-1]
    leftupadd = integralImage[x-(h+1)][y-(h+1)]
    leftup = int(leftup1) - int(leftup2) - int(leftup3) + int(leftupadd)
    
    rightup1 = integralImage[x][y+h]
    rightup2 = leftup1
    rightup3 = integralImage[x-(h+1)][y+h]
    rightupadd = leftup3
    rightup = int(rightup1) - int(rightup2) - int(rightup3) + int(rightupadd)
    
    leftdown1 = integralImage[x+h][y-1]
    leftdown2 = integralImage[x+h][y-(h+1)]
    leftdown3 = leftup1
    leftdownadd = leftup2
    leftdown = int(leftdown1) - int(leftdown2) - int(leftdown3) + int(leftdownadd)
    
    rightdown1 = integralImage[x+h][y+h]
    rightdown2 = leftdown1
    rightdown3 = rightup1
    rightdownadd = leftup1
    rightdown = int(rightdown1) - int(rightdown2) - int(rightdown3) + int(rightdownadd)
    
    #print('Ada Test 4: Left Up: ', leftup, ', Right Up: ', rightup, ', Left Down: ', leftdown, ', Right Down: ', rightdown)
    return leftup, rightup, leftdown, rightdown