import cv2

I = cv2.imread('rice.png', cv2.IMREAD_GRAYSCALE)
J = cv2.blur(I,(5,5))
ret, BJ= cv2.threshold(J, 120, 255, type=cv2.THRESH_BINARY)
cv2.imshow('image',BJ)
cv2.waitKey(0)

