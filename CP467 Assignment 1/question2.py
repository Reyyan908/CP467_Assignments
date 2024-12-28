import cv2

# 1: Negative of the image
def image_negative(image):
    height = image.shape[0]
    width = image.shape[1]
    negative_image = image.copy()
    for i in range(height):
        for j in range(width):
            negative_image[i, j] = 255 - int(image[i, j])
    return negative_image

# 2: Power-law (Gamma) transformation
def power_law_transformation(image, gamma):
    height = image.shape[0]
    width = image.shape[1]
    power_image = image.copy()
    for i in range(height):
        for j in range(width):
            normalized_pixel = int(image[i, j]) / 255.0
            transformed_pixel = normalized_pixel ** gamma
            new_value = int(transformed_pixel * 255)
            if new_value > 255:
                new_value = 255
            elif new_value < 0:
                new_value = 0
            power_image[i, j] = new_value
    return power_image

# 3: Bit-plane slicing
def bit_plane_slicing(image):
    height = image.shape[0]
    width = image.shape[1]
    bit_planes = []
    for bit in range(8):
        bit_plane_image = image.copy()
        for i in range(height):
            for j in range(width):
                pixel_value = int(image[i, j])
                bit_value = (pixel_value >> bit) & 1
                bit_plane_image[i, j] = bit_value * 255
        bit_planes.append(bit_plane_image)
    return bit_planes

def main():
    image = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image not found or unable to load.")
        return

    #1: Negative of the image
    negative_image = image_negative(image)
    cv2.imwrite('cameraman_negative.tif', negative_image)

    # 2: Power-law transformation
    gamma = 0.5  
    power_image = power_law_transformation(image, gamma)
    cv2.imwrite('cameraman_power.tif', power_image)

    # 3: Bit-plane slicing
    bit_planes = bit_plane_slicing(image)
    for idx, bit_plane_image in enumerate(bit_planes):
        cv2.imwrite(f'cameraman_b{idx+1}.tif', bit_plane_image)

if __name__ == '__main__':
    main()