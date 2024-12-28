import cv2
import numpy as np

# Read + error handle
inputImage = cv2.imread('lena.tif', cv2.IMREAD_GRAYSCALE)

if inputImage is None:
    print("Error: Could not read the image.")
    exit()

# Image info
rows, cols = inputImage.shape
kernelSize = 7
sigma = 1.0
mean = 0.0

ax = np.linspace(-(kernelSize - 1) / 2., (kernelSize - 1) / 2., kernelSize)
xx, yy = np.meshgrid(ax, ax)

# Gaussian function
kernel = np.exp(-0.5 * (np.square(xx - mean) + np.square(yy - mean)) / np.square(sigma))
kernel = kernel / np.sum(kernel)  

padSize = kernelSize // 2
paddedImage = np.pad(inputImage, pad_width=padSize, mode='constant', constant_values=0)

outputImage = np.zeros_like(inputImage, dtype=np.float32)

for i in range(rows):
    for j in range(cols):
        # Extract 
        roi = paddedImage[i:i+kernelSize, j:j+kernelSize]
        output_value = np.sum(roi * kernel)
        outputImage[i, j] = output_value

outputImage = np.clip(outputImage, 0, 255)
outputImage = outputImage.astype(np.uint8)

cv2.imwrite('t1b.tif', outputImage)