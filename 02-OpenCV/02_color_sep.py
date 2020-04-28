import cv2
img = cv2.imread('lena.png')

blue, green, red = cv2.split(img)

cv2.imshow('canal R',red)

cv2.waitKey(0)
cv2.destroyAllWindows()
