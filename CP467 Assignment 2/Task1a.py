import cv2
import numpy as np

# Read + error handle
inputImage = cv2.imread('lena.tif', cv2.IMREAD_GRAYSCALE)

if inputImage is None:
    print("Error: Could not read the image.")
    exit()

rows, cols = inputImage.shape
kernel = np.ones((3, 3), dtype=np.float32) / 9.0
paddedImage = np.pad(inputImage, pad_width=1, mode='constant', constant_values=0)

outputImage = np.zeros_like(inputImage)

for i in range(rows):
    for j in range(cols):
        # Extract 3x3
        roi = paddedImage[i:i+3, j:j+3]
        outputValue = np.sum(roi * kernel)
        outputImage[i, j] = outputValue

cv2.imwrite('t1a.tif', outputImage)