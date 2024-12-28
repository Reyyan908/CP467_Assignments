import cv2
import numpy as np

# Load + error handle
inputImage = cv2.imread('lena.tif', cv2.IMREAD_GRAYSCALE)

if inputImage is None:
    print("Error: Could not read the image.")
    exit()

# Task 3a
kernelSize = (7, 7)  
sigma = 1.0 
gaussianImage = cv2.GaussianBlur(inputImage, ksize=kernelSize, sigmaX=sigma)
laplacianImage = cv2.Laplacian(gaussianImage, cv2.CV_64F)

laplacianImage = np.absolute(laplacianImage)  
laplacianImage = (laplacianImage / np.max(laplacianImage)) * 255
laplacianImage = laplacianImage.astype(np.uint8)

cv2.imwrite('t3a.tif', laplacianImage)

# Task 3b
Lthreshold = 100  
Uthreshold = 200
cannyImage = cv2.Canny(inputImage, Lthreshold, Uthreshold)

cv2.imwrite('t3b.tif', cannyImage)