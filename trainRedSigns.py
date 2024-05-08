import violajones
import cv2
import numpy as np
import pandas as pd
import featureselector
    
# This function is used to extract traffic sign features and classify them. This returns two DataFrames: the 4 Haar Features of
# each feature, and the classification DataFrame.
def trainSigns(size = 400):
    
    train = pd.DataFrame(columns=['Aba Boost 1', 'Aba Boost 2', 'Aba Boost 3', 'Aba Boost 4'])
    signs = pd.DataFrame(columns=['Signs'])
    
    with open("redsigns.txt", "r") as file:
        for i in file:
            fileNameArr = i.rstrip().split(" ")
            #print(fileNameArr)
            signName = "signs/" + fileNameArr[0] + ".png"
            
            image = cv2.imread(signName) 
            image = cv2.resize(image, (size, size))
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            imageWithFeatures, features = featureselector.selectFeaturesFromImage(image)
            integralImage = violajones.calculateIntegralImage(image)
            
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                featuresInSigns = []
                area = cv2.contourArea(contour)
                if area > 10000:
                    #print('Area: ', area)
                    
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
                            #cv2.circle(image, (y, x), point_size, (255, 0, 0), -1)
            
                    ababoostfeatures = violajones.computeHaarFeatures(integralImage, featuresInSigns, gray_image)
                    
                    train = pd.concat([train, ababoostfeatures], ignore_index=True)
                    #print(signName)
                    #print(ababoostfeatures)
                
                    count = int(fileNameArr[1])
                
                    df = pd.DataFrame({'Signs': [int(count)] * len(ababoostfeatures)})
                    signs = pd.concat([signs, df], ignore_index=True)
                    #cv2.imshow("Shapes Detected", image)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    #print(signs)
        train.to_csv('trainRed.csv', index=False)
        signs.to_csv('trainClassifierRed.csv', index=False)
    
def trainRed():
    trainSigns()
    print('Done With Training And Classifying Red Signs. Check directory for .csv files.')