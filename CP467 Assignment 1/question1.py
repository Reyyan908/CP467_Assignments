import cv2
import math

# Read image
original_img = cv2.imread('lena.tif')
height, width = original_img.shape[:2]

new_height, new_width = height // 2, width // 2
downscaled_img = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

def calculate_mse(original, compared):
    height, width, channels = original.shape
    mse = 0
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                diff = int(original[i, j][c]) - int(compared[i, j][c])
                mse += diff * diff
    mse /= (height * width * channels)
    return mse

# 1: Nearest Neighbor Interpolation from Scratch 
nearest_scratch = original_img.copy()
nearest_scratch[:] = 0  

scale_x = new_width / width
scale_y = new_height / height

for i in range(height):
    for j in range(width):
        x = (j + 0.5) * scale_x - 0.5
        y = (i + 0.5) * scale_y - 0.5

        x = int(round(x))
        y = int(round(y))

        x = max(0, min(x, new_width - 1))
        y = max(0, min(y, new_height - 1))

        nearest_scratch[i, j] = downscaled_img[y, x]

cv2.imwrite('lena_nearest_scratch.tif', nearest_scratch)

mse_nearest_scratch = calculate_mse(original_img, nearest_scratch)
print(f"MSE for Nearest Neighbor Interpolation (from scratch): {mse_nearest_scratch}")

# 2: Nearest Neighbor Interpolation using OpenCV
nearest_cv = cv2.resize(downscaled_img, (width, height), interpolation=cv2.INTER_NEAREST)
cv2.imwrite('lena_nearest_cv.tif', nearest_cv)

mse_nearest_cv = calculate_mse(original_img, nearest_cv)
print(f"MSE for Nearest Neighbor Interpolation (OpenCV): {mse_nearest_cv}")

# 3: Bilinear Interpolation from Scratch 
bilinear_scratch = original_img.copy()
bilinear_scratch[:] = 0  

scale_x = new_width / width
scale_y = new_height / height

for i in range(height):
    for j in range(width):
        x = (j + 0.5) * scale_x - 0.5
        y = (i + 0.5) * scale_y - 0.5

        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1

        x0 = max(0, min(x0, new_width - 1))
        x1 = max(0, min(x1, new_width - 1))
        y0 = max(0, min(y0, new_height - 1))
        y1 = max(0, min(y1, new_height - 1))

        dx = x - x0
        dy = y - y0

        for c in range(3):
            I00 = int(downscaled_img[y0, x0][c])
            I10 = int(downscaled_img[y0, x1][c])
            I01 = int(downscaled_img[y1, x0][c])
            I11 = int(downscaled_img[y1, x1][c])

            value = (I00 * (1 - dx) * (1 - dy) +
                     I10 * dx * (1 - dy) +
                     I01 * (1 - dx) * dy +
                     I11 * dx * dy)

            value = int(round(value))
            value = max(0, min(255, value))

            bilinear_scratch[i, j][c] = value

cv2.imwrite('lena_bilinear_scratch.tif', bilinear_scratch)

mse_bilinear_scratch = calculate_mse(original_img, bilinear_scratch)
print(f"MSE for Bilinear Interpolation (from scratch): {mse_bilinear_scratch}")

# 4: Bilinear Interpolation using OpenCV
bilinear_cv = cv2.resize(downscaled_img, (width, height), interpolation=cv2.INTER_LINEAR)
cv2.imwrite('lena_bilinear_cv.tif', bilinear_cv)

mse_bilinear_cv = calculate_mse(original_img, bilinear_cv)
print(f"MSE for Bilinear Interpolation (OpenCV): {mse_bilinear_cv}")

# 5: Bicubic Interpolation using OpenCV
bicubic_cv = cv2.resize(downscaled_img, (width, height), interpolation=cv2.INTER_CUBIC)
cv2.imwrite('lena_bicubic_cv.tif', bicubic_cv)

mse_bicubic_cv = calculate_mse(original_img, bicubic_cv)
print(f"MSE for Bicubic Interpolation (OpenCV): {mse_bicubic_cv}")