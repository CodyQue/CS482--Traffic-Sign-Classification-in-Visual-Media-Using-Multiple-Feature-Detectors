import violajones
import featureselector
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

testFeatureIndex = 0
testClassification = None # Size = Number of unique signs
signNames = ['No Turn On Red', 'No U-Turn']

def cosineSimilarity(row):
    global testClassification
    global testFeatureIndex
    
    #print(testClassification)
    # Opens the train.csv and trainClassifier.csv files (obtained from the trainSigns.py program)
    train = pd.read_csv('train.csv')
    signs = pd.read_csv('trainClassifier.csv')

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


# Function to display image in UI
def displayImage(image):
    cv2.imshow('haris_corner', image) 
    cv2.waitKey() 
    
# Main function that imports image and selects features from image
def main():
    global testClassification
    #image = cv2.imread("yieldsigns/yield.jfif") 
    #image = cv2.imread("yieldsigns/yield3.jpg") 
    #image = cv2.imread("stopsigns/stopsign.jfif") 
    #image = cv2.imread("stopsigns/stopsign3.jpg") 
    #image = cv2.imread("speedsigns/speed.png") 
    #image = cv2.imread("signs/stop&yield.jpg") 
    image = cv2.imread("signs/yield.png") 
    image = cv2.imread("signs/stopsign3.png") 
    #image = cv2.imread("yieldsigns/yield3.jpg") 
    #image = cv2.imread("signs/noturnonred2.png") 
    image = cv2.imread("signs/stop&yield.jpg") 
    
    image = cv2.resize(image, (400, 400))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(gray_image)
    #displayImage(gray_image)
    
    imageWithFeatures, features = featureselector.selectFeaturesFromImage(image)
    #displayImage(imageWithFeatures)
    
    # Gets array of signs for classifier count
    signs = pd.read_csv('trainClassifier.csv')
    uniqueSigns = signs['Signs'].unique()
    testClassification = np.zeros(uniqueSigns.shape[0])
    #print(features)
    
    # Calculates integral image
    integralImage = violajones.calculateIntegralImage(image)
    
    ababoostfeatures = violajones.computeHaarFeatures(integralImage, features, gray_image)
    ababoostfeatures.apply(cosineSimilarity, axis=1)
    #print(ababoostfeatures)
    
    #print('Min: ', ababoostfeatures.min())
    #print('Max: ', ababoostfeatures.max())
    #print('Mean: ', ababoostfeatures.mean())
    
    #derivative(gray_image)
    
    displayImage(imageWithFeatures)
    
    #blurred, edges = canny_edge_detection(image) 
    #displayImage(edges)
    
if __name__ == "__main__":
    main()