import cv2

img = cv2.imread('rice.png')
laplace = cv2.Laplacian(img,cv2.CV_8U,ksize=15)


uSharp= cv2.subtract(img, laplace)
cv2.imshow('img',img)
cv2.imshow('usharp',uSharp)
cv2.waitKey(0)
