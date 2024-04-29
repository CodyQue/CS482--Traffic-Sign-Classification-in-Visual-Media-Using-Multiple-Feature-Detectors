import violajones
import featureselector
import cv2
import numpy as np

def derivative(image):
    # Read the image

    # Apply Sobel filter for x-direction derivative
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

    # Apply Sobel filter for y-direction derivative
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Convert back to uint8
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # Display the results
    cv2.imshow('Sobel X', sobel_x)
    cv2.imshow('Sobel Y', sobel_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def canny_edge_detection(frame): 
    # Convert the frame to grayscale for edge detection 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
      
    # Apply Gaussian blur to reduce noise and smoothen edges 
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5) 
      
    # Perform Canny edge detection 
    edges = cv2.Canny(blurred, 70, 135) 
      
    return blurred, edges

def showFeaturesInBlack(image, features):
    for i in features:
        x,y = i
        print(i)
        image[x, y] = [0, 0, 0]
        cv2.imshow('Colored Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Function to display image in UI
def displayImage(image):
    cv2.imshow('haris_corner', image) 
    cv2.waitKey() 
    
# Main function that imports image and selects features from image
def main():
    #image = cv2.imread("yieldsigns/yield.jfif") 
    #image = cv2.imread("yieldsigns/yield3.jpg") 
    image = cv2.imread("stopsigns/stopsign.jfif") 
    #image = cv2.imread("stopsigns/stopsign3.jpg") 
    #image = cv2.imread("speedsigns/speed.png") 
    #image = cv2.imread("signs/stop&yield.jpg") 
    image = cv2.resize(image, (400, 400))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(gray_image)
    #displayImage(gray_image)
    
    imageWithFeatures, features = featureselector.selectFeaturesFromImage(image)
    #displayImage(imageWithFeatures)
        
    #print(features)
    integralImage = violajones.calculateIntegralImage(image)
    
    ababoostfeatures = violajones.computeHaarFeatures(integralImage, features, gray_image)
    print(ababoostfeatures)
    
    print('Min: ', ababoostfeatures.min())
    print('Max: ', ababoostfeatures.max())
    print('Mean: ', ababoostfeatures.mean())
    
    #derivative(gray_image)
    
    displayImage(imageWithFeatures)
    
    #blurred, edges = canny_edge_detection(image) 
    #displayImage(edges)
    
if __name__ == "__main__":
    main()