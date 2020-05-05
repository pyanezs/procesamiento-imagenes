import cv2
from skimage import filters

img = cv2.imread('circuit.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges= filters.sobel(gray)
out = cv2.normalize(edges.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('Sobel',out)
cv2.waitKey(0)
