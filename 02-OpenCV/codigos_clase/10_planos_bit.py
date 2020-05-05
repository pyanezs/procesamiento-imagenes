import cv2
import numpy as np

# Read image
img = cv2.imread(
    "/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/cameraman.png")

# Select channel 0
img = img[:,:,0]
for k in range(0, 8):
    plane = np.full((img.shape[0], img.shape[1]), 2 ** k, np.uint8)
    res = plane & img
    x = res * 255

    cv2.imshow("plane", x)
    cv2.waitKey(0)
