import cv2

img = cv2.imread('cameraman.png')
img_median = cv2.blur(img,(7,7))
cv2.imshow('Imagen Mediana',img_median)
cv2.waitKey(0)
