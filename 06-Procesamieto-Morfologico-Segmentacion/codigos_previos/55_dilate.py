import cv2
import numpy as np

img = cv2.imread('j.png')

# MORPH_CROSS, MORPH_ELLIPSE, MORPH_RECT
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
print(kernel)
dilate = cv2.dilate(img,kernel)

cv2.imshow('Dilatacion',img)
cv2.waitKey(0)

