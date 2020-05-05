
import cv2

import matplotlib.pyplot as plt

img = cv2.imread("/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/lena.png")


hist_blue = cv2.calcHist([img], [0], None, [256], [0,256])
plt.plot(hist_blue)
plt.xlim([0,256])
plt.show()

hist_green = cv2.calcHist([img], [1], None, [256], [0, 256])
plt.plot(hist_green)
plt.xlim([0, 256])
plt.show()


hist_red = cv2.calcHist([img], [2], None, [256], [0, 256])
plt.plot(hist_red)
plt.xlim([0, 256])
plt.show()
