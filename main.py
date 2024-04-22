import violajones
import featureselector
import cv2
import numpy as np

# Function to display image in UI
def displayImage(image):
    cv2.imshow('haris_corner', image) 
    cv2.waitKey() 
    
# Main function that imports image and selects features from image
def main():
    image = cv2.imread("yield2.jpg") 
    image = cv2.resize(image, (400, 400))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(gray_image)
    
    imageWithFeatures, features = featureselector.selectFeaturesFromImage(image)
    #print(features)
    integralImage = violajones.calculateIntegralImage(image)
    
    #displayImage(imageWithFeatures)
    
    violajones.computeHaarFeatures(integralImage, features)
    
if __name__ == "__main__":
    main()