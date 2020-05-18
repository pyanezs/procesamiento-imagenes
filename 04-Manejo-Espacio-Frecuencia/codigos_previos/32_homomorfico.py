import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

img = cv2.imread('rice.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
F = np.fft.fft2(gray)
FS = np.fft.fftshift(F)

m = 256
YH = 0.9
YL = 0.2
c  = 0.005
vector = np.linspace(-m,m, m)
X, Y = np.meshgrid(vector,vector)

Z = (YH-YL) * (1-np.exp(-c* (np.power(X,2)+np.power(Y,2)) ))+YL

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.Spectral)
plt.show()

BF = Z * FS

S= np.fft.ifft2(np.fft.fftshift(BF))
out = cv2.normalize(abs(S), None, 0.0, 1.0, cv2.NORM_MINMAX)

cv2.imshow('Salida', out)
cv2.waitKey(0)