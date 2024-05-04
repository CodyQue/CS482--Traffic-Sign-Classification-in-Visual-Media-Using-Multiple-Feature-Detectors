import violajones
import featureselector
import cv2
import numpy as np

#image = cv2.imread("signs/stop&yield.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("yieldsigns/yield.jfif", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread("stopsigns/stopsign3.jpg", cv2.IMREAD_GRAYSCALE) 
#image = cv2.imread("yieldsigns/yield3.jpg", cv2.IMREAD_GRAYSCALE) 

image = cv2.resize(image, (400, 400))

params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 10000

# Filter by Area
params.filterByArea = True
params.minArea = 100

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create Blob Detector
detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(image)

image_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Blobs Detected", image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()