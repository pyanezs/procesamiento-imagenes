import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/cameraman.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

r = 60
c = 30

x = np.linspace(-127, 128, 256)
y = np.linspace(-127, 128, 256)
X, Y = np.meshgrid(x, y)

Z1 = np.sqrt(np.power(X, 2)+np.power(Y, 2)) > r-c
Z2 = np.sqrt(np.power(X, 2)+np.power(Y, 2)) < r+c

Z = np.bitwise_and(Z1, Z2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z)
plt.show()

F = np.fft.fft2(gray)
FS = np.fft.fftshift(F)
BF = FS * Z

S = np.fft.ifft2(np.fft.fftshift(BF))
out = cv2.normalize(abs(S), None, 0.0, 1.0, cv2.NORM_MINMAX)

cv2.imshow('S', out)
cv2.waitKey(0)
