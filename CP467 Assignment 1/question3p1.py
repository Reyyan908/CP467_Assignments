import cv2
import numpy as np

img = cv2.imread('einstein.tif', 0)

flat = img.flatten()
hist, bins = np.histogram(flat, 256, [0, 256])

cdf = hist.cumsum()

cdf_masked = np.ma.masked_equal(cdf, 0)  
cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
cdf_final = np.ma.filled(cdf_normalized, 0).astype('uint8')  

img_equalized = cdf_final[img]

cv2.imwrite('einstein_equalized.tif', img_equalized)