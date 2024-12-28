import cv2
import numpy as np
import os

# image list
image_names = ["iris1.tif", "iris2.tif", "iris3.tif", "iris4.tif", "iris5.tif"]

# directories to save edge map and output images
def create_output_dirs():
    os.makedirs('Edge_Maps', exist_ok=True)
    os.makedirs('Output_Images', exist_ok=True)

# enhance contrast to improve edge visibility 
def enhance_contrast(image, clip_limit=3.0):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(image)

# process pupil and iris boundaries
def process_image(image_name):
    img_path = os.path.join('Input_Images', image_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Image {image_name} not found in Input_Images folder.")
        return

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # enhance contrast for pupil detection
    enhanced_gray_pupil = enhance_contrast(gray, clip_limit=3.0)

    # creating a binary image where pupil becomes black 
    _, thresholded_pupil = cv2.threshold(enhanced_gray_pupil, 50, 255, cv2.THRESH_BINARY_INV)

    # filling for small holes in pupil region
    kernel_pupil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_pupil = cv2.morphologyEx(thresholded_pupil, cv2.MORPH_CLOSE, kernel_pupil)

    # edge detection for pupil
    edges_pupil = cv2.Canny(closed_pupil, 40, 120)

    # pupil edge map
    pupil_edge_map_name = f"{os.path.splitext(image_name)[0]}_pupil_edge.tif"
    cv2.imwrite(os.path.join('Edge_Maps', pupil_edge_map_name), edges_pupil)

    # circle boundaries in pupil edge map  
    pupil_circles = cv2.HoughCircles(
        edges_pupil,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,
        param1=100,
        param2=20,
        minRadius=15,
        maxRadius=50
    )

    # if pupil detected, draw for output image
    if pupil_circles is not None:
        pupil_circles = np.int32(np.around(pupil_circles[0, :]))
        pupil_circle = pupil_circles[0] # first detected circle
        pupil_center = (pupil_circle[0], pupil_circle[1])
        pupil_radius = pupil_circle[2]
        cv2.circle(img, pupil_center, pupil_radius, (0, 255, 0), 2)  # green circle for pupil

        # expected radius for iris based off pupil size
        min_iris_radius = int(pupil_radius * 1.8)
        max_iris_radius = int(pupil_radius * 2.8)

        # enhance contrast for iris detection
        enhanced_gray_iris = enhance_contrast(gray, clip_limit=5.0)

        # gaussian blur to reduce noise
        blurred_iris = cv2.GaussianBlur(enhanced_gray_iris, (9, 9), 0)

        # edge detection for iris
        edges_iris = cv2.Canny(blurred_iris, 50, 150)

        # mask centered from iris to sclera 
        mask = np.zeros_like(edges_iris)
        cv2.circle(mask, pupil_center, max_iris_radius, 255, thickness=-1)
        masked_edges = cv2.bitwise_and(edges_iris, edges_iris, mask=mask)

        # iris edge map
        iris_edge_map_name = f"{os.path.splitext(image_name)[0]}_iris_edge.tif"
        cv2.imwrite(os.path.join('Edge_Maps', iris_edge_map_name), masked_edges)

        # circle boundaries detected in iris edge map 
        param2_value = 20
        iris_circles = cv2.HoughCircles(
            masked_edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=param2_value,
            minRadius=min_iris_radius,
            maxRadius=max_iris_radius
        )

        if iris_circles is not None:
            iris_circles = np.int32(np.around(iris_circles[0, :]))

            # fine tuning to detect best circle match
            expected_iris_radius = int(pupil_radius * 2.3)
            weight_radius = 2.5
            weight_center = 4.0

            # scoring on detected iris circles based on distance from pupil center and radius difference
            scores = []
            for circle in iris_circles:
                center = (circle[0], circle[1])
                radius = circle[2]
                dx = center[0] - pupil_center[0]
                dy = center[1] - pupil_center[1]
                distance = np.hypot(dx, dy)
                radius_diff = abs(radius - expected_iris_radius)
                score = weight_center * distance + weight_radius * radius_diff
                scores.append((score, circle))

            # sort by score and select the best circle
            scores.sort(key=lambda x: x[0])
            best_circle = scores[0][1]
            cv2.circle(img, (best_circle[0], best_circle[1]), best_circle[2], (255, 0, 0), 2)  # blue circle for iris
        else:
            print(f"No suitable iris circle detected in image {image_name}.")
    else:
        print(f"No suitable pupil circle detected in image {image_name}.")

    # combine pupil and iris edge maps
    combined_edges = cv2.bitwise_or(edges_pupil, masked_edges)
    combined_edge_map_name = f"{os.path.splitext(image_name)[0]}_combined_edge.tif"
    cv2.imwrite(os.path.join('Edge_Maps', combined_edge_map_name), combined_edges)

    # output image for both boundaries
    output_image_name = f"{os.path.splitext(image_name)[0]}_output.tif"
    cv2.imwrite(os.path.join('Output_Images', output_image_name), img)
    print(f"Processed and saved output for {image_name}.")

def main():
    create_output_dirs()
    for image_name in image_names:
        process_image(image_name)

if __name__ == "__main__":
    main()