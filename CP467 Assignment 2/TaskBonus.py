import time
import numpy as np
import cv2
import pandas as pd

# Read + error handle 
image = cv2.imread('lena.tif', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not read the image.")
    exit()

# Execution time
def measureTime(taskFunc, *args):
    startTime = time.time()
    taskFunc(*args)
    endTime = time.time()
    return endTime - startTime

# Task 1a
def myAveragingFilter(image):
    rows, cols = image.shape
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    paddedImage = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    outputImage = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            roi = paddedImage[i:i + 3, j:j + 3]
            outputValue = np.sum(roi * kernel)
            outputImage[i, j] = outputValue
    return outputImage

# Task 1b
def myGaussianFilter(image):
    rows, cols = image.shape
    kernelSize = 7
    sigma = 1.0
    mean = 0.0
    ax = np.linspace(-(kernelSize - 1) / 2., (kernelSize - 1) / 2., kernelSize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx - mean) + np.square(yy - mean)) / np.square(sigma))
    kernel = kernel / np.sum(kernel)
    padSize = kernelSize // 2
    paddedImage = np.pad(image, pad_width=padSize, mode='constant', constant_values=0)
    outputImage = np.zeros_like(image, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            roi = paddedImage[i:i + kernelSize, j:j + kernelSize]
            outputValue = np.sum(roi * kernel)
            outputImage[i, j] = outputValue
    return np.clip(outputImage, 0, 255).astype(np.uint8)

# Task 1c
def mySobelFilter(image):
    rows, cols = image.shape
    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    paddedImage = np.pad(image, pad_width=1, mode='edge')
    outputImageX = np.zeros_like(image, dtype=np.float32)
    outputImageY = np.zeros_like(image, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            roi = paddedImage[i:i + 3, j:j + 3]
            outputImageX[i, j] = np.sum(roi * sobelX)
            outputImageY[i, j] = np.sum(roi * sobelY)
    gradientMagnitude = np.sqrt(np.square(outputImageX) + np.square(outputImageY))
    return (gradientMagnitude / np.max(gradientMagnitude) * 255).astype(np.uint8)

# Task 3a
def myMarrHildreth(image):
    gaussianImage = cv2.GaussianBlur(image, (7, 7), 1.0)
    laplacianImage = cv2.Laplacian(gaussianImage, cv2.CV_64F)
    return np.absolute(laplacianImage).astype(np.uint8)

# Task 3b
def myCanny(image):
    return cv2.Canny(image, 100, 200)

# Scratch implementations
timeMyAvg = measureTime(myAveragingFilter, image)
timeMyGauss = measureTime(myGaussianFilter, image)
timeMySobel = measureTime(mySobelFilter, image)
timeMyMarr = measureTime(myMarrHildreth, image)
timeMyCanny = measureTime(myCanny, image)

# OpenCV implementations
timeOpencvAvg = measureTime(cv2.blur, image, (3, 3))
timeOpencvGauss = measureTime(cv2.GaussianBlur, image, (7, 7), 1.0)
timeOpencvSobel = measureTime(cv2.Sobel, image, cv2.CV_64F, 1, 0) + measureTime(cv2.Sobel, image, cv2.CV_64F, 0, 1)
timeOpencvMarr = measureTime(cv2.GaussianBlur, image, (7, 7), 1.0) + measureTime(cv2.Laplacian, image, cv2.CV_64F)
timeOpencvCanny = measureTime(cv2.Canny, image, 100, 200)

# Table
data = {
    "Task": ["T1: Averaging Filter", "T1: Gaussian Filter", "T1: Sobel Filter", "T3: Marr-Hildreth", "T3: Canny"],
    "Implementation (s)": [timeMyAvg, timeMyGauss, timeMySobel, timeMyMarr, timeMyCanny],
    "OpenCV (s)": [timeOpencvAvg, timeOpencvGauss, timeOpencvSobel, timeOpencvMarr, timeOpencvCanny],
    "Difference (s)": [
        timeMyAvg - timeOpencvAvg,
        timeMyGauss - timeOpencvGauss,
        timeMySobel - timeOpencvSobel,
        timeMyMarr - timeOpencvMarr,
        timeMyCanny - timeOpencvCanny
    ]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))