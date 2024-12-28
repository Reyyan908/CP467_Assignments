import numpy as np
import cv2

def connectedComponentLabeling(edgeImage):
    rows, cols = edgeImage.shape
    labels = np.zeros((rows, cols), dtype=np.int32) 
    label = 1  
    parent = {}  

    for i in range(rows):
        for j in range(cols):
            if edgeImage[i, j] == 0:
                continue 
            neighbors = []
            if i > 0:
                if edgeImage[i - 1, j] != 0:
                    neighbors.append(labels[i - 1, j])  
                if j > 0 and edgeImage[i - 1, j - 1] != 0:
                    neighbors.append(labels[i - 1, j - 1])  
                if j < cols - 1 and edgeImage[i - 1, j + 1] != 0:
                    neighbors.append(labels[i - 1, j + 1])  
            if j > 0 and edgeImage[i, j - 1] != 0:
                neighbors.append(labels[i, j - 1])  

            if not neighbors:
                labels[i, j] = label
                parent[label] = label  
                label += 1
            else:
                minLabel = min(neighbors)
                labels[i, j] = minLabel
                for lbl in neighbors:
                    if lbl != minLabel:
                        union(parent, lbl, minLabel)

    labelMapping = {}
    for lbl in range(1, label):
        root = find(parent, lbl)
        labelMapping[lbl] = root

    for i in range(rows):
        for j in range(cols):
            if labels[i, j] != 0:
                labels[i, j] = labelMapping[labels[i, j]]

    uniqueLabels = np.unique(labels)
    uniqueLabels = uniqueLabels[uniqueLabels != 0]  
    numLabels = len(uniqueLabels)
    labelToColor = {}
    
    np.random.seed(0)  
    colors = np.random.randint(0, 255, size=(numLabels, 3))
    for idx, lbl in enumerate(uniqueLabels):
        labelToColor[lbl] = colors[idx]

    outputImage = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            lbl = labels[i, j]
            if lbl != 0:
                outputImage[i, j] = labelToColor[lbl]

    return outputImage

def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, x, y):
    xRoot = find(parent, x)
    yRoot = find(parent, y)
    if xRoot != yRoot:
        parent[yRoot] = xRoot  

def applyThreshold(image, thresholdValue):
    binaryImage = np.where(image > thresholdValue, 255, 0)
    return binaryImage.astype(np.uint8)

edgeImageA = cv2.imread('t3a.tif', cv2.IMREAD_GRAYSCALE)
if edgeImageA is None:
    print("Error: Could not read the edge image t3a.tif")
    exit()

edgeBinaryA = applyThreshold(edgeImageA, thresholdValue=50)

outputImageA = connectedComponentLabeling(edgeBinaryA)
cv2.imwrite('t4a.tif', outputImageA)