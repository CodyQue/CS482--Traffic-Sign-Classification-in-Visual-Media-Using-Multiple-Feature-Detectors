# Traffic Sign Classifier Using The Viola-Jones Algorithm
By Cody Querubin, Misha Burnayev, Alexander Walker

## Problem/Motivation
Implementing machine vision in a vehicular setting has been rapidly growing and increasing, especially with technologies such as self-driving cars and traffic cameras. Although there are many training models for machine visions, such as YOLOS, SSDs, CNNs, R-CNNs, and many more, this project’s objective is to explore a different, and potentially even faster, method for classifying data in computer visions and compare the results to these training models. We wanted to pick an algorithm that is uncommonly used to detect traffic signs; therefore, the main algorithm used in this project to identify traffic signs is the Viola-Jones algorithm. This algorithm is mainly used for face detection; however, this project will alter the training to be able to classify traffic signs, instead of faces.

Aside from using the Viola-Jones algorithm, other techniques such as feature selection and cosine similarity were also used to help detect and train images to detect traffic signs. The library, OpenCV, was used for selecting features in images and selecting shapes. The library, sklearn, was also used for classifying test images.

## The Viola-Jones Algorithm

### Algorithm
Proposed in 2001 by Paul Viola and Michael Jones, the Viola-Jones algorithm is a machine-learning algorithm used for object detection in images and videos. The most common practice of using this algorithm is for face detection. This section explains how the Viola-Jone algorithm works, and how this algorithm is altered to detect and classify traffic signs. The procedure is as follows: it first computes the Integral Image; then using the Integral Image, it computes the Haar Features of the image, and the classification of signs depends on the outcome of these Haar Features.

### The Integral Image
The algorithm first converts the image into a grayscale image. This is to make sure each pixel value is between 0 and 255; it would be very complicated if the pixel values were in RGB values. The algorithm's second step is the integral image, the sum of all pixels at a specific image row and column. Once the entire integral image is computed, the algorithm loops through every possible feature in the image. It first finds the total number of pixels of, and around, the feature. An example below demonstrates how the total number of pixels is determined.

#### Efficiency
The reason this machine-learning algorithm works well and fast is because of the utilization of the Integral Image, which saves a lot of time when computing the total number of pixels of a specific area. Rather than taking the pixel sum by looping through each pixel of a specific area, which takes O(n^2) time complexity, it determines the sum of all pixels of a specific feature by 1) Getting the sum of all pixels of a specific point, 2) Subtracts the total number of sums of the bottom left, 3) Subtracts the total number of sums of the upper right, 4) Adds the total number of pixels of the upper left. This alone is a time complexity of O(1).

### Haar Features
Haar Features are essential to the Viola-Jones algorithm since these are features to determine object recognition. These features are measured by taking the difference between two different areas of a selected feature in the image. However, what makes Haar Features reliable are the different types of features and the different ways of computing these Haar Features. Although there are many different Haar Features, this project utilizes the four main different Haar Features: the x-axis edge feature, the y-axis edge feature, the line feature, and the four-rectangle feature. These are what the different Haar Features look like, respectively.

### Implementation
The entire Viola-Jones algorithm, calculating the integral image and computing the Haar Features, used for this project was developed in the violajones.py. The algorithm was all developed, with the help of a few sources to understand how the algorithm worked.

## Utilizing OpenCV For Feature Selection And Shape Detection
Traditionally, the Haar Features are determined by looping through the entire image to measure pixel intensity patterns. However, for this project, the program uses the OpenCV library to extract major features from the image. 

Additionally, shapes were also detected using the OpenCV library. This is done by finding the Contours and looping every Coutours to find ‘big shapes’. These big shapes are assumed to be traffic signs. This is important.

Both feature and shape selection are essential for this project. Combining these two algorithms would enable the selection of features inside of signs. It is essential to make sure it selects actual sign features instead of selecting a random feature, like a feature from a tree or a house since these extra features could contribute to the wrong classification.

## Training/Preprocessing Data
The training is accomplished through three separate programs, testSigns.py, testRedSigns.py, and testYellowSigns.py. This is used to train signs based on sign color; this will be explained in the Classifying Signs In Images section as to why this is relevant. Although they train different signs, the algorithm to train is the same. The program, training.py is used to call the trainSigns.py, trainRedSigns.py, and trainYellowSigns.py and train and classify signs.

The first thing done is to open the .txt file (signs.txt, redsigns.txt, and yellowsigns.txt depending on the color classification). The .txt file(s) contain the name of the file name and their classification number. The training program loops through every row of the .txt file and opens the image file.

It then detects the ‘shape’ of the sign and extracts features inside it using the OpenCV library. Once it gathers all of the features inside of the sign, it computes the Viola-Jones algorithm on every feature in the sign, which returns values of each Haar Feature. These Haar Features are stored in their respective .csv file (the CSV files with the name train____.csv); this will be opened when classifying test images. The classifications are also stored in their respective CSV files (files with the name trainClassifier____.csv).

#### Data Storage
A separate directory, named input, is used to store the training set of all of the signs. When running the training program, it selects the sign images inside of the directory. To add to the training set, add the image to this directory, and add the sign to the .txt file, depending on what color the sign is.

#### Exceptions
Some signs do not have a training set since they are uniquely shaped. For example, stop signs are the only octagon-shaped signs, and pedestrian/school crossing signs are the only pentagon-shaped signs. Therefore, there are no training sets for these signs, and instead automatically classify octagons as stop signs and pentagons as pedestrian/school crossing signs.

## Classifying Signs In Images
The user inputs images, and the program will classify the main signs inside the image. Just like how the data is trained, the program detects the ‘’shape’ of the sign and extracts features inside of the sign; then it computes the Viola-Jones algorithm on every feature in the sign.

#### Color Classification
Once each Haar Feature is calculated in the image, it moves on to classifying the sign. It first determines the color of the sign; this is where the sign color, from the Training Data, comes in. If the test sign is red, then it only considers red signs when classifying. The same logic applies to yellow and black/white signs. This is to make sure it does not loop through every single sign feature and instead considers signs of the same color.

#### Sign Classification Using Cosine Similarity
Classifying signs in images utilizes the Cosine Similarity algorithm, from the sklearn library, to find the closest and most similar train feature. Once it finds the closest feature, the feature where the cosine similarity value is greater than 0.95, it assigns that sign classification to the test feature. After performing cosine similarity and classifying all of the test features, it picks the classification with the greatest count, and that sign is classified.

## Problems And Future Improvements

#### Amount Of Signs That Exist
One small issue is the large number of signs that exist today. The program classifies the few, common, important signs of the United States, such as stop signs, yield signs, no right turn, no turn or red, windy roads, speed limits, and one-way. However, there are still a lot more signs that do exist.

Aside from the signs in the United States, there are also signs to consider from other countries around the world, and these signs differ from signs in the United States depending on sign color, pixel intensity patterns, and many more.

#### Blurry/Low-Quality Images Containing Traffic Signs
There is one major issue when running the program. Although the program works with images clearly showing the sign, it does not work for blurry/distorted images containing signs. However, it is not the program developed that is causing this issue. The reason is because of OpenCV’s shape detection; this tool makes it difficult to detect shapes on low-quality images.

There are possible solutions to fix this; the easy solution is the use blob detection, instead of shape detection and select the ‘big blobs’ of the image, assuming these big blobs contain the signs. One solution, if wanting to continue with shape detection, is using interpolation to approximate signs in images. The issue using this is the implementation complexity; it would need to predict the color of the sign. This would be complicated to implement but would work well to make objects look clearer.

#### Objects Blocking Signs
Another objective the OpenCV’s shape detection does not work well on is detecting shapes with objects in the way of it. The bottom image of the stop sign demonstrates this. Because the tree branch is blocking the stop sign, the OpenCV’s shape detection cannot detect the shape of the sign being an octagon.

## Tutorial/How To Run The Program
Before running the program, these dependencies need to be installed:
- Cv2
- Pandas
- NumPy
- sklearn

#### Running Train Data
The directory already contains CSV files containing the Haar Features and the classification file. However, if wanting to add signs to the training set, add the image to the signs directory and edit the .txt file, depending on the color of the sign. Run the training.py program to run all of the training programs (trainSigns.py, trainRedSigns.py, and trainYellowSigns.py). 

#### Running Classifying Program
The first thing to do is put images in the input directory; the program loops through every image in the directory. Run the main.py program to start classifying signs in the image. Most of the computations, such as color and sign classification and shape detection, are done through the main.py program. Additionally, other programs, such as the violajones.py and featureselector.py programs are also called upon from main.py.
