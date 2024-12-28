import cv2
import numpy as np

# Load + error handle
inputImage = cv2.imread('lena.tif', cv2.IMREAD_GRAYSCALE)

if inputImage is None:
    print("Error: Could not read the image.")
    exit()

# Task 2a
kernelSize1 = (3, 3)
averagedImage = cv2.blur(inputImage, ksize=kernelSize1)
cv2.imwrite('t2a.tif', averagedImage)

# Task 2b
kernelSize2 = (7, 7)
sigma = 1.0
gaussianImage = cv2.GaussianBlur(inputImage, ksize=kernelSize2, sigmaX=sigma)
cv2.imwrite('t2b.tif', gaussianImage)

# Task 2c
kernelSize3 = 3

# Gradients
sobelX = cv2.Sobel(inputImage, cv2.CV_64F, dx=1, dy=0, ksize=kernelSize3)
sobelY = cv2.Sobel(inputImage, cv2.CV_64F, dx=0, dy=1, ksize=kernelSize3)

sobelCombined = np.sqrt(np.square(sobelX) + np.square(sobelY))
sobelCombined = (sobelCombined / np.max(sobelCombined)) * 255
sobelCombined = sobelCombined.astype(np.uint8)

cv2.imwrite('t2c.tif', sobelCombined)