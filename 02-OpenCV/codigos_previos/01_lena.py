import cv2

import matplotlib.pyplot as plt

img = cv2.imread('lena.png')

blue, green, red = cv2.split(img)

cv2.imwrite('canal_azul.png', blue)

hist_red = cv2.calcHist([img],[2],None,[256],[0,256])
plt.plot(hist_red)
plt.xlim([0,256])
plt.show()
