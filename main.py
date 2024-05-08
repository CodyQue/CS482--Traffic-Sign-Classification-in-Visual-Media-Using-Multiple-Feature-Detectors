import violajones
import featureselector
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

testFeatureIndex = 0
testClassification = None # Size = Number of unique signs
signNamesRed = ['No Turn On Red', 'No U-Turn', 'Do Not Enter']
signNamesYellow = ['Speed Limit', 'Windy Roads']
signNames = ['No Turn On Red','Speed Limit', 'One Way']

# This is used to classify new images. It takes in a feature, from the inputted image, and performs cosine similarity.
# It then finds the train data that is the most similar to it
def cosineSimilarity(row, trainName, signsName):
    global testClassification
    global testFeatureIndex
    
    #print("GOING THROUGH COSINE SIMILARITY")
    #print(testClassification)
    # Opens the train.csv and trainClassifier.csv files (obtained from the trainSigns.py program)
    train = pd.read_csv(trainName)
    signs = pd.read_csv(signsName)
    
    uniqueSigns = signs['Signs'].unique()
    
    # Converts DataFrames into NumPy arrays.
    rowNP = np.array(row).reshape(1, -1)
    trainNP = np.array(train)
    
    #print(trainNP)
    #print(rowNP)
    cosSimArr = cosine_similarity(rowNP, trainNP)
    
    if (cosSimArr[0][np.argmax(cosSimArr)] >= 0.97):
        indexOrder = np.argsort(cosSimArr)
        indexOrder = indexOrder[0][::-1]
        index = indexOrder[0]
        signClassifier = (signs.iloc[index]['Signs'])
        #print(signClassifier)
        for i in range(len(uniqueSigns)):
            if uniqueSigns[i] == signClassifier:
                testClassification[i] += 1
        testFeatureIndex += 1

# This function gathers the features inside each identified shape. The program first identifies every shape in the
# image and finds the biggest shapes (this is to assume that these big shapes are the traffic signs). Then it finds
# the features inside of the shapes using OpenCV's pointPolygonTest. Once it detects the features inside of the signs,
# then it determines the sign using the Viola-Jones algorithm.
#
# NOTE: ChatGPT was used to help determine shapes inside of the image and to determine the unique colors inside of the shape.

def gatherFeaturesInShape(image, gray_image, features, integralImage):
    global testClassification
    global signNames
    
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detects every shape in the image.
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    isSign = False
    areaStart = 10000
    
    while (isSign != True):
        # Loops through every shape in the image.
        for contour in contours:
            featuresInSigns = [] # List to store features inside of the shapes.
            area = cv2.contourArea(contour)
            if area > areaStart: # This is used to determine if the contour is a 'big shape'.
                epsilon = 0.03 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Selects features inside of the big shape
                point_size=1
                for f in features:
                    x, y = f
                    tuplePoint = (float(f[1]), float(f[0]))
                    determine = cv2.pointPolygonTest(contour, tuplePoint, measureDist=False)
                    if (determine >= 0):
                        featuresInSigns.append(f)
                        #cv2.circle(image, (y, x), point_size, (255, 0, 0), -1) # Used to plot the features inside of the sign.
                        
                featuresInSigns = np.array(featuresInSigns)
                #print("Features: ", features)
                #print("New Features: ", featuresInSigns)
                #print('Done')
                
                # Determines the color of the sign. 
                mask = np.zeros_like(image)
                cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                shape_colors = cv2.bitwise_and(image, mask)
                color_list = shape_colors.reshape(-1, 3)
                unique_colors = np.unique(color_list, axis=0)
                        
                yellow_present = False
                red_present = False
                redCount = 0
                yellowCount = 0
                    
                #print(color_list)
                        
                for color in color_list:
                    # Check if the color is red
                    #print("Color: ", color)
                    if (color[2] > 120) and (color[1] < 50) and (color[0] < 100) and (red_present == False):
                        #print('There is red: ', color)
                        red_present = True
                        redCount += 1
                    # Checks if the color is yellow
                    elif (color[2] > 120) and (color[1] > 120) and (color[0] < 100) and yellow_present == False:
                        #print('There is yellow: ', color)
                        yellow_present = True
                        yellowCount += 1
                    
                #print("Red: ", redCount, ", Yellow: ", yellowCount)

                num_vertices = len(approx)
                signName = "unknown"
                
                # Since stop signs are the only octagon signs, this automatically becomes a stop sign
                if (num_vertices == 8 or num_vertices == 7) and (red_present == True): 
                    signName = "Stop Sign"
                
                # Since Pedestrian Crossings are the only pentagon-shaped signs, this automatically becomes a Pedestrian Crossings sign
                elif num_vertices == 5 and (yellow_present == True):
                    signName = "Pedestrian Crossing"
                    
                elif num_vertices == 3:
                    signName = "Yield"
                
                else:
                    # Computes the Viola-Jones algorithm
                    ababoostfeatures = violajones.computeHaarFeatures(integralImage, featuresInSigns, gray_image)
                    
                    if red_present == True:
                        # Creates array for classifying test data
                        #print('Test Sign is Red')
                        signs = pd.read_csv('trainClassifierRed.csv')
                        uniqueSigns = signs['Signs'].unique()
                        testClassification = np.zeros(uniqueSigns.shape[0])
                        #print('Unique: ', uniqueSigns)
                        
                        trainName = 'trainRed.csv'
                        signsName = 'trainClassifierRed.csv'
                        ababoostfeatures.apply(lambda x: cosineSimilarity(x,trainName,signsName), axis=1) # Uses KNN (Cosine Similarity) to classify the sign.
                    
                    # This is for the yellow signs. If the sign is yellow, then it predicts based on other yellow signs.
                    elif yellow_present == True:
                        # Creates array for classifying test data
                        #print('Test Sign is Yellow')
                        signs = pd.read_csv('trainClassifierYellow.csv')
                        uniqueSigns = signs['Signs'].unique()
                        testClassification = np.zeros(uniqueSigns.shape[0])
                        #print('Unique: ', uniqueSigns)
                        
                        trainName = 'trainYellow.csv'
                        signsName = 'trainClassifierYellow.csv'
                        ababoostfeatures.apply(lambda x: cosineSimilarity(x,trainName,signsName), axis=1) # Uses KNN (Cosine Similarity) to classify the sign.
                    
                    # This is for signs without the color red or yellow.
                    else:
                        # Creates array for classifying test data
                        #print('White & Black')
                        signs = pd.read_csv('trainClassifier.csv')
                        uniqueSigns = signs['Signs'].unique()
                        testClassification = np.zeros(uniqueSigns.shape[0]) 
                        #print('Unique: ', uniqueSigns)
                        
                        trainName = 'train.csv'
                        signsName = 'trainClassifier.csv'
                        ababoostfeatures.apply(lambda x: cosineSimilarity(x,trainName,signsName), axis=1) # Uses KNN (Cosine Similarity) to classify the sign.
                
                    #print(ababoostfeatures)
                    
                    #print("Done With Cosine Similarity: ", testClassification)

                    largestSign = 0
                    for i in range(len(testClassification)):
                        if red_present == True:
                            if testClassification[i] > largestSign:
                                signName = signNamesRed[i]
                                largestSign = testClassification[i]
                        elif yellow_present == True:
                            if testClassification[i] > largestSign:
                                signName = signNamesYellow[i]
                                largestSign = testClassification[i]
                        else:
                            if testClassification[i] > largestSign:
                                signName = signNames[i]
                                largestSign = testClassification[i]
                    
                # Outlines the 'sign' in the image
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
                cv2.putText(image, signName, (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print("Sign: ", signName)
                isSign = True
            
        if isSign == False:
            #print('IsSign is False')
            areaStart -= 2000
            if areaStart == 0:
                break
   
    #image = cv2.resize(image, (400, 400))
    cv2.imshow("Shapes Detected", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Main function that imports image and selects features from image
def main(arguments):
    global testClassification
    folder_path = "input/"

    # Loop through each file in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is an image (you may need to adjust the condition based on your file extensions)
        if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Construct the full path to the image file
            image_file = os.path.join(folder_path, file_name)
            print(image_file)  # Or perform any other operation with the image file
    
            try:
                image = cv2.imread(image_file) 
                
                image = cv2.resize(image, (400, 400))
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #print(gray_image)
                #displayImage(gray_image)
                
                imageWithFeatures, features = featureselector.selectFeaturesFromImage(image)
                #displayImage(imageWithFeatures)
                    
                #print('Features: ', features)
                integralImage = violajones.calculateIntegralImage(image)
                gatherFeaturesInShape(image, gray_image, features, integralImage)
                
            except Exception as e:
                print("File Does Not Exist, Please Try Again")
    exit()
    
if __name__ == "__main__":
    arguments = sys.argv[1:]
    main(arguments)