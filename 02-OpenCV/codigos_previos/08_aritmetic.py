import cv2

lena= cv2.imread('lena.png')
gray1 = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
print(gray1.shape)

barbara= cv2.imread('barbara_gray.bmp')
gray2 =  cv2.cvtColor(barbara, cv2.COLOR_BGR2GRAY)
print(gray2.shape)

output = cv2.addWeighted(gray1,0.3, gray2, 0.7, 0)

cv2.imshow('output', output)
cv2.waitKey(0)

