import cv2
import numpy as np

# Load the image
image = cv2.imread("shapes2.png") 
#image = cv2.resize(image, (500, 500))

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 100

# Filter by Circularity
params.filterByCircularity = False

# Filter by Convexity
params.filterByConvexity = False

# Filter by Inertia
params.filterByInertia = False

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(image)

# Draw detected blobs as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the detected blobs
cv2.imshow("Blobs", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()