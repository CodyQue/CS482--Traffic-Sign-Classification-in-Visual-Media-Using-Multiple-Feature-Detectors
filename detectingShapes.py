import cv2
import numpy as np

# Load the image
image = cv2.imread("stopsigntest.jpg") 
image = cv2.resize(image, (500, 500))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours
for contour in contours:
    # Approximate the contour to reduce the number of points
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Determine the shape of the contour based on the number of vertices
    num_vertices = len(approx)
    shape = "unidentified"
    
    if num_vertices == 6:
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
        shape = "hexagon"
        cv2.putText(image, shape, (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


# Display the image with detected shapes
cv2.imshow("Shapes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()