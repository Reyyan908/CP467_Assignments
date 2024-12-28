import cv2
import numpy as np

def dfs(image, visited, x, y, label, component):
    rows, cols = image.shape
    stack = [(x, y)]
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),        (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    while stack:
        cx, cy = stack.pop()
        if visited[cx, cy] or image[cx, cy] == 0:
            continue
        visited[cx, cy] = True
        component[cx, cy] = label
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                stack.append((nx, ny))

# Find connected components
def findConnectedComponents(image):
    rows, cols = image.shape
    visited = np.zeros((rows, cols), dtype=bool)
    component = np.zeros((rows, cols), dtype=int)
    label = 1
    for x in range(rows):
        for y in range(cols):
            if image[x, y] > 0 and not visited[x, y]:
                dfs(image, visited, x, y, label, component)
                label += 1
    return component, label - 1

# Color connected components
def colorComponents(componentMap, numComponents):
    rows, cols = componentMap.shape
    outputImage = np.zeros((rows, cols, 3), dtype=np.uint8)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(numComponents + 1, 3))
    for x in range(rows):
        for y in range(cols):
            label = componentMap[x, y]
            if label > 0:
                outputImage[x, y] = colors[label]
    return outputImage


cannyImage = cv2.imread('t3b.tif', cv2.IMREAD_GRAYSCALE)

componentsCanny, numComponentsCanny = findConnectedComponents(cannyImage)
coloredCanny = colorComponents(componentsCanny, numComponentsCanny)

cv2.imwrite('t4b.tif', coloredCanny)