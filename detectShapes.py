import cv2
import numpy as np

# Read the image
image = cv2.imread('input/donotenter.png')
print(image)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection using Canny
edges = cv2.Canny(gray, 50, 150)

# Find contours in the edged image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours
for contour in contours:
    # Approximate the contour
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Draw the contour on the original image
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    
# Display the result
cv2.imshow('Shapes Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()