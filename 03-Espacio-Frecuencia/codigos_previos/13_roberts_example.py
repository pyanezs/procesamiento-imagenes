import cv2
from skimage import filters

img = cv2.imread('cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

roberts = filters.roberts(gray)
cv2.imshow('roberts',roberts)
cv2.waitKey(0)
