import cv2

img = cv2.imread('cameraman.png')
imgBlur = cv2.blur(img,(5,5))
cv2.imshow('image',imgBlur)
cv2.waitKey(0)

