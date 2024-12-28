import cv2
import numpy as np

source_img = cv2.imread('chest_x-ray1.jpeg', 0)
reference_img = cv2.imread('chest_x-ray2.jpeg', 0)

source_hist, bins = np.histogram(source_img.flatten(), 256, [0, 256])
reference_hist, bins = np.histogram(reference_img.flatten(), 256, [0, 256])

source_cdf = source_hist.cumsum()
reference_cdf = reference_hist.cumsum()

source_cdf_masked = np.ma.masked_equal(source_cdf, 0)
reference_cdf_masked = np.ma.masked_equal(reference_cdf, 0)

source_cdf_normalized = (source_cdf_masked - source_cdf_masked.min()) * 255 / (source_cdf_masked.max() - source_cdf_masked.min())
reference_cdf_normalized = (reference_cdf_masked - reference_cdf_masked.min()) * 255 / (reference_cdf_masked.max() - reference_cdf_masked.min())

source_cdf = np.ma.filled(source_cdf_normalized, 0).astype('uint8')
reference_cdf = np.ma.filled(reference_cdf_normalized, 0).astype('uint8')

mapping = np.zeros(256, dtype='uint8')

for src_pixel_val in range(256):
    src_cdf_val = source_cdf[src_pixel_val]
    diff = np.abs(reference_cdf - src_cdf_val)
    closest_pixel_val = np.argmin(diff)
    mapping[src_pixel_val] = closest_pixel_val

matched_img = mapping[source_img]

cv2.imwrite('chest_x-ray3.jpeg', matched_img)