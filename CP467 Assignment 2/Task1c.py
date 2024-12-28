import cv2
import numpy as np

# Read + error handle
inputImage = cv2.imread('lena.tif', cv2.IMREAD_GRAYSCALE)

if inputImage is None:
    print("Error: Could not read the image.")
    exit()

rows, cols = inputImage.shape

sobelX = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobelY = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

paddedImage = np.pad(inputImage, pad_width=1, mode='edge')

outputImageX = np.zeros_like(inputImage, dtype=np.float32)
outputImageY = np.zeros_like(inputImage, dtype=np.float32)
outputImage = np.zeros_like(inputImage, dtype=np.float32)

# Sobel X kernel
for i in range(rows):
    for j in range(cols):
        # Extract 3x3
        roi = paddedImage[i:i+3, j:j+3]
        gx = np.sum(roi * sobelX)
        outputImageX[i, j] = gx

# Sobel Y kernel
for i in range(rows):
    for j in range(cols):
        # Extract 3x3
        roi = paddedImage[i:i+3, j:j+3]
        gy = np.sum(roi * sobelY)
        outputImageY[i, j] = gy

gradientMagnitude = np.sqrt(np.square(outputImageX) + np.square(outputImageY))
gradientMagnitude = (gradientMagnitude / np.max(gradientMagnitude)) * 255.0

outputImage = gradientMagnitude.astype(np.uint8)
cv2.imwrite('t1c.tif', outputImage)