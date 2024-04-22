import violajones
import featureselector
import cv2
import numpy as np
  
def canny_edge_detection(frame): 
    # Convert the frame to grayscale for edge detection 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
      
    # Apply Gaussian blur to reduce noise and smoothen edges 
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5) 
      
    # Perform Canny edge detection 
    edges = cv2.Canny(blurred, 70, 135) 
      
    return blurred, edges

# Function to display image in UI
def displayImage(image):
    cv2.imshow('haris_corner', image) 
    cv2.waitKey() 
    
# Main function that imports image and selects features from image
def main():
    image = cv2.imread("yieldsigns/yield.jfif") 
    #image = cv2.imread("stopsigns/stopsign3.jpg") 
    image = cv2.resize(image, (400, 400))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(gray_image)
    #displayImage(gray_image)
    
    imageWithFeatures, features = featureselector.selectFeaturesFromImage(image)
    #print(features)
    integralImage = violajones.calculateIntegralImage(image)
    
    violajones.computeHaarFeatures(integralImage, features)
    
    displayImage(imageWithFeatures)
    
    #blurred, edges = canny_edge_detection(image) 
    #displayImage(edges)
    
if __name__ == "__main__":
    main()