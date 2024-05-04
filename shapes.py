import cv2
import numpy as np

# Read the input image
#image = cv2.imread("signs/noturnonred.png") 
#image = cv2.imread("signs/stopsign3.png") 
#image = cv2.imread("signs/stopsign4.png")
image = cv2.imread("signs/stop&yield.jpg")  
#image = cv2.imread("signs/yield2.png")  

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image = cv2.resize(image, (400, 400))
# Apply GaussianBlur to remove noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = None
largest_area = -1
shape = None

# Loop over contours
for contour in contours:
    area = cv2.contourArea(contour)
    
    if area > 10000:
        print('Area: ', area)
        largest_contour = contour
        largest_area = area
        # Approximate the contour
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Determine shape based on number of vertices
        num_vertices = len(approx)
        shape = "unknown"
        if num_vertices == 3:
            shape = "triangle"
        elif num_vertices == 4:
            # Check if it's a square or rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            shape = "square" if aspect_ratio >= 0.95 and aspect_ratio <= 1.05 else "rectangle"
        elif num_vertices == 5:
            shape = "pentagon"
        elif num_vertices == 8:
            shape = "Stop Sign"
        else:
            shape = "circle"
            # Draw contours and shape name
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
#cv2.putText(image, shape, (largest_contour.ravel()[0], largest_contour.ravel()[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
# Display the result
image = cv2.resize(image, (400, 400))
cv2.imshow("Shapes Detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()