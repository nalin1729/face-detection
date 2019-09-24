import numpy as np
from numpy import sqrt
import imageio

# Parameters of the Project

sampleSize = 40 # Number of Data Points for Training
width = 300 # Width in Pixels of each Training Image
height = 399 # Height in Pixels of each Training Image
thetaThreshold = 300 # The threshold determining at which point to reject the image

# Returns the euclidean norm of a vector vec
def euclidMag(vec):
    mag = 0
    for i in range(vec.size):
        mag += vec[i] ** 2
        
    return sqrt(mag)

# Given the mean, std. deviation, and an element of the set, determine if
# the element is an outlier
def isOutlier(mean, stdev, elem):
    z = abs((mean - elem) / stdev)
    if z >= 2: # 2 deviations from the mean will be considered an outlier
        return True
    return False
    
# Finds the minimum value of a dists that is not an outlier
def findSmallestNonOutlier(dists):
    np.sort(dists)
    
    stdev = np.std(dists)
    smallest = max(dists)
    mean = np.mean(dists)
    for i in range (dists.size):
        if dists[i] < smallest and not isOutlier(mean, stdev, dists[i]):
            smallest = dists[i]
    
    return smallest
    
# Reads in all precalculated information (as calculated by calculate.py)
def readData():
    omega = np.loadtxt("Trained_Data/omega.txt")
    mean = np.loadtxt("Trained_Data/mean.txt")
    usefulEigFaces = np.loadtxt("Trained_Data/eigen.txt")
    std = np.loadtxt("Trained_Data/std.txt")
    
    return usefulEigFaces, omega, mean, std
        
                    
if __name__ == '__main__':
    usefulEigFaces, omega, mean, std = readData()
    
    # Reads in a test image (can change this line of code to test different images)
    img = imageio.imread("Test_Images_Random/test10.jpg")
    aux = np.empty([height, width])
    for j in range(0, height):
        for k in range(0, width):
            aux[j][k] =  0.2989*img[j][k][0] + 0.5870*img[j][k][1] + 0.1140*img[j][k][2]
            
    # Calculate the characteristic vector of the test image
    img = np.asarray(aux).reshape(-1)
    w = (usefulEigFaces.T).dot(np.divide(img - mean, std))
    
    dists = np.empty([sampleSize])
    for i in range(0, sampleSize):
        dists[i] = euclidMag(w - omega[:,i])
        
    minDist = findSmallestNonOutlier(dists)
    
    # to view the distance of the image from the eigenface space, uncomment line 72
    print("Distance from data set: " + str(minDist))
            
    if (minDist <= thetaThreshold):
        print("Algorithm determines that this is a human face")
    else: 
        print("Algorithm determines that this is NOT a human face")
