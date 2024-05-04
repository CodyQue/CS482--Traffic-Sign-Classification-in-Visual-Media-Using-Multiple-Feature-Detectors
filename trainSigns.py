import violajones
import featureselector
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

train = None # For storing all of the training sign data into a dataframe
signs = None # Classifiers for signs
oneCount = 0 #Stop Sign
twoCount = 0 # Yield Sign
testArr = []
index1 = 0
features = None

def cosineSimilarity(row):
    global oneCount
    global twoCount
    global index1
    global features
    global testArr
    rowNP = np.array(row).reshape(1, -1)
    trainNP = np.array(train)
    cosSimArr = cosine_similarity(rowNP, trainNP)
    
    if (cosSimArr[0][np.argmax(cosSimArr)] >= 0.97):
        indexOrder = np.argsort(cosSimArr)
        indexOrder = indexOrder[0][::-1]
        #print(indexOrder)
        #print(cosSimArr[0][1605])
        
        signArr = []
        for i in range(11):
            index = indexOrder[i]
            signArr.append(signs.iloc[index]['Signs'])
        print(signArr)
        testArr.append(features[index1])
        index1 += 1
    
def showFeaturesInBlack(image, features, point_size=1):
    for feature in features:
        x, y = feature
        cv2.circle(image, (y, x), point_size, (0, 255, 0), -1)  # -1 indicates filled circle
    cv2.imshow('Colored Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def selectFeaturesFromImage(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(gray_image)
    gray_image = np.float32(gray_image) 

    dst = cv2.cornerHarris(gray_image, blockSize=2, ksize=5, k=0.04) 
    
    feature_coords = np.argwhere(dst > 0.01 * dst.max())

    #print('Harris Corner: ', dst)

    dst = cv2.dilate(dst, None) 
    #image[dst > 0.01 * dst.max()] = [0, 255, 0] 
    return image, feature_coords
    

def trainSigns():
    global train
    global signs
    global features
    global testArr
    #image = cv2.imread("yieldsigns/yield.jfif") 
    #image = cv2.imread("yieldsigns/yield3.jpg") 
    #image = cv2.imread("stopsigns/stopsign.jfif") 
    #image = cv2.imread("stopsigns/stopsign3.jpg") 
    #image = cv2.imread("speedsigns/speed.png") 
    #image = cv2.imread("signs/stop&yield.jpg") 
    
    train = pd.DataFrame(columns=['Aba Boost 1', 'Aba Boost 2', 'Aba Boost 3', 'Aba Boost 4'])
    signs = pd.DataFrame(columns=['Signs'])
    
    with open("signs.txt", "r") as file:
        for i in file:
            fileNameArr = i.rstrip().split(" ")
            #print(fileNameArr)
            signName = "signs/" + fileNameArr[0] + ".png"
            
            image = cv2.imread(signName) 
            image = cv2.resize(image, (400, 400))
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            imageWithFeatures, features = featureselector.selectFeaturesFromImage(image)
            integralImage = violajones.calculateIntegralImage(image)
            ababoostfeatures = violajones.computeHaarFeatures(integralImage, features, gray_image)
            train = pd.concat([train, ababoostfeatures], ignore_index=True)
            #print(signName)
            #print(ababoostfeatures)
            count = int(fileNameArr[1])
            df = pd.DataFrame({'Signs': [int(count)] * len(ababoostfeatures)})
            signs = pd.concat([signs, df], ignore_index=True)
    print(train)
    print(signs)
    
    #image = cv2.imread("stopsigns/stopsign3.jpg") 
    #image = cv2.imread("yieldsigns/yield3.jpg") 
    #image = cv2.imread("signs/yield.png") 
    #image = cv2.imread("signs/stopsign4.png") 
    #image = cv2.imread("signs/stop&yield.jpg") 
    image = cv2.imread("signs/noturnonred2.png") 
    
    image = cv2.resize(image, (400, 400))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageWithFeatures, features = featureselector.selectFeaturesFromImage(image)
    integralImage = violajones.calculateIntegralImage(image)
    test = violajones.computeHaarFeatures(integralImage, features, gray_image)
    closest = test.apply(cosineSimilarity, axis=1)
    print('Stop Count: ', oneCount)
    print('Yield Count: ', twoCount)
    #print(index1)
    testArr = np.array(testArr)
    #print(testArr)
    showFeaturesInBlack(image, testArr)
    
trainSigns()