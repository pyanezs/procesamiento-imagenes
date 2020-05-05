import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

img = cv2.imread('cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

#vamos a crear ruido
V = np.linspace(0,1,num=256)
Y= 0.2*np.sin(2*np.pi*10*V)
M= np.matlib.repmat(Y,256,1)
noise_img= np.add(M, gray)
cv2.imshow('signal', noise_img)


#vamos aplicar fourier
F = np.fft.fft2(noise_img)
fshift = np.fft.fftshift(F)

#bloqueamos las frecuencias con ruido
fshift[128:130,118:120]=0.0
fshift[128:130,138:140]=0.0

S= np.fft.ifft2(np.fft.fftshift(fshift))
cv2.imshow('Ouput', abs(S))
cv2.waitKey(0)