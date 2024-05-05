import violajones
import featureselector
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

testFeatureIndex = 0
testClassification = None # Size = Number of unique signs
signNames = ['No Turn On Red', 'No U-Turn']

# This is used to classify new images. It takes in a feature, from the inputted image, and performs cosine similarity.
# It then finds the train data that is the most similar to it
def cosineSimilarity(row):
    global testClassification
    global testFeatureIndex
    
    #print(testClassification)
    # Opens the train.csv and trainClassifier.csv files (obtained from the trainSigns.py program)
    train = pd.read_csv('train.csv')
    signs = pd.read_csv('trainClassifier.csv')
    
    uniqueSigns = signs['Signs'].unique()
    #print(uniqueSigns)
    
    # Converts DataFrames into NumPy arrays.
    rowNP = np.array(row).reshape(1, -1)
    trainNP = np.array(train)
    
    #print(trainNP)
    cosSimArr = cosine_similarity(rowNP, trainNP)
    
    if (cosSimArr[0][np.argmax(cosSimArr)] >= 0.97):
        indexOrder = np.argsort(cosSimArr)
        indexOrder = indexOrder[0][::-1]
        index = indexOrder[0]
        index2 = (signs.iloc[index]['Signs'])-1
        #print(index2)
        testClassification[index2] += 1
        

        testFeatureIndex += 1

# This function gathers the features inside each identified shape. The program first identifies every shape in the
# image and finds the biggest shapes (this is to assume that these big shapes are the traffic signs). Then it finds
# the features inside of the shapes using OpenCV's pointPolygonTest. Once it detects the features inside of the signs,
# then it determines the sign using the Viola-Jones algorithm.
def gatherFeaturesInShape(image, gray_image, features, integralImage):
    global testClassification
    global signNames
    
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detects every shape in the image.
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loops through every shape in the image.
    for contour in contours:
        featuresInSigns = [] # List to store features inside of the shapes.
        area = cv2.contourArea(contour)
        if area > 10000: # This is used to determine if the contour is a 'big shape'.
            
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
                    cv2.circle(image, (y, x), point_size, (255, 0, 0), -1)
                    
            featuresInSigns = np.array(featuresInSigns)
            #print("Features: ", features)
            #print("New Features: ", featuresInSigns)
            #print('Done')

            num_vertices = len(approx)
            signName = "unknown"
            
            # Since stop signs are the only octagon signs, this automatically becomes a stop sign
            if num_vertices == 8 or num_vertices == 7: 
                signName = "Stop Sign"
            
            # Since Pedestrian Crossings are the only pentagon-shaped signs, this automatically becomes a Pedestrian Crossings sign
            elif num_vertices == 5:
                signName = "Pedestrian Crossing"
                
            elif num_vertices == 3:
                signName = "Yield"
            
            else:
                ababoostfeatures = violajones.computeHaarFeatures(integralImage, featuresInSigns, gray_image)
                #print(ababoostfeatures)
                ababoostfeatures.apply(cosineSimilarity, axis=1)
                
                largestSign = 0
                for i in range(len(testClassification)):
                    if testClassification[i] > largestSign:
                        signName = signNames[i]
                        largestSign = testClassification[i]
                
             # Outlines the 'sign' in the image
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.putText(image, signName, (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
   
    #image = cv2.resize(image, (400, 400))
    cv2.imshow("Shapes Detected", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Main function that imports image and selects features from image
def main():
    global testClassification
    #image = cv2.imread("signs/stop&yield.jpg") 
    #image = cv2.imread("signs/yield.png") 
    #image = cv2.imread("signs/stopsign3.png") 
    #image = cv2.imread("yieldsigns/yield3.jpg") 
    #image = cv2.imread("signs/noturnonred2.png") 
    #image = cv2.imread("signs/stop&yield.jpg") 
    #image = cv2.imread("signs/pedestriancrossing.png") 
    #image = cv2.imread("signs/stopsign2.png") 
    #image = cv2.imread("signs/yield2.png") 
    #image = cv2.imread("signs/pedestriancrossing.png") 
    #image = cv2.imread("signs/noparking.png") 
    #image = cv2.imread("signs/noturnonred.png") 
    #image = cv2.imread("signs/yield3.png") 
    #image = cv2.imread("signs/stopsign5.png")
    image = cv2.imread("signs/stopsign6.png")
    #image = cv2.imread("signs/yield4.png") 
    #image = cv2.imread("signs/nouturn6.png") 
    #image = cv2.imread("signs/noturnonred3.png") 
    
    image = cv2.resize(image, (800, 800))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(gray_image)
    #displayImage(gray_image)
    
    imageWithFeatures, features = featureselector.selectFeaturesFromImage(image)
    
    # Creates array for classifying test data
    signs = pd.read_csv('trainClassifier.csv')
    uniqueSigns = signs['Signs'].unique()
    testClassification = np.zeros(uniqueSigns.shape[0])
    #displayImage(imageWithFeatures)
        
    #print('Features: ', features)
    integralImage = violajones.calculateIntegralImage(image)
    featuresInShape = gatherFeaturesInShape(image, gray_image, features, integralImage)
    print(testClassification)
    
if __name__ == "__main__":
    main()