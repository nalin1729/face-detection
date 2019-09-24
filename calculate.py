import numpy as np
from numpy import sqrt
import imageio
from numpy import linalg as LA

# Parameters of the Project

sampleSize = 40 # Number of Data Points for Training
width = 300 # Width in Pixels of each Training Image
height = 399 # Height in Pixels of each Training Image
threshold = 10 # The number of eigenfaces to actually be used

# The information calculated from the training set, to be saved in files for testing

mean = np.empty([height * width]) 
usefulEigFaces = np.empty([width * height, threshold])
omega = np.empty([threshold, sampleSize]) # Characteristic Vectors of each Data Point
stdDev = np.empty([height * width]) # Standard Deviation of each Random Variable,
                                    # used to normalize data measurements

# Returns a unit vector in the direction of vec
                                    
def norm(vec):
    mag = 0
    for i in range(vec.size):
        mag += vec[i] ** 2
        
    mag = sqrt(mag)
    
    for i in range(vec.size):
        vec[i] /= mag
        
    return vec

# Reads in training data from subdirectory Training_Data, calculates the eigenfaces
# and takes stores the most important of them as given by a threshold. Also calculates
# and stores the characteristic vectors of each data point
def calculateEigenFaces():
    
    # Load the Images of the Training Set
    a = np.empty([width * height, sampleSize]) # each image
    aux = np.empty([height, width])
    for i in range(1, sampleSize + 1):
        img = imageio.imread("Training_Data/IMG" + str(int(i / 10)) + str(i % 10) + ".jpg")
        for j in range(0, height):
            for k in range(0, width):
                aux[j][k] =  0.2989*img[j][k][0] + 0.5870*img[j][k][1] + 0.1140*img[j][k][2]
        a[:,i - 1] = np.asarray(aux).reshape(-1)
        
    # Treating each pixel as an independent random variable, calculate the std. deviation
    for i in range(height * width):
        stdDev[i] = np.std(a[i,:])
    
    # Calculate the mean face from the data set
    for i in range(height * width):
        mean[i] = np.mean(a[i,:])
    
    # Calculates the matrix F, which stores the standardized data points
    diff = np.empty([height * width,  sampleSize])
    for i in range(sampleSize):
        diff[:,i] = np.divide((a[:,i] - mean), stdDev)
    
    # Calculate the eigenvectors and eigenvalues of the matrix F^t F
    mat = (diff.T).dot(diff)
    eigVals, eigVecs = LA.eigh(mat)
    eigenFaces = np.empty([width * height, sampleSize])
    
    # Using the heuristic, calculate the eigenfaces (Fv, where v are the
    # eigenvectors of F^t F) of the data set
    eigenFaces = diff.dot(eigVecs)
    #for i in range(sampleSize):
    #    for k in range(sampleSize):
    #        eigenFaces[:,i] += eigVecs[k][i] * diff[:,k]
            
    # Normalize the eigenfaces to be of unit length
    for i in range(sampleSize):
        eigenFaces[:,i] = norm(eigenFaces[:,i])
    
    # Select the most important eigenfaces, given by a threshold
    
    sortedEigVals = np.sort(eigVals)
    
    for i in range(1, threshold + 1):
        for j in range(sampleSize):
            if (sortedEigVals[-1 * i] == eigVals[j]):
                usefulEigFaces[:,i - 1] = eigenFaces[:,j]
   
    # Calculates the characteristic vector of each data point
    for i in range(0, sampleSize):
        omega[:,i] = (usefulEigFaces.T).dot(diff[:,i])
        
# Saves all important calculated data to a file
def writeData():
    np.savetxt("Trained_Data/omega.txt", omega, delimiter=' ')
    np.savetxt("Trained_Data/mean.txt", mean, delimiter=' ')
    np.savetxt("Trained_Data/eigen.txt", usefulEigFaces, delimiter=' ')
    np.savetxt("Trained_Data/std.txt", stdDev, delimiter=' ')
    
# Main method
if __name__ == '__main__':
    calculateEigenFaces()
    writeData()
