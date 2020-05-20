import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm

def ruido_turbulento(f,k):

    F=np.fft.fft2(f)
    m,n =F.shape

    rx = np.linspace(-m/2, m/2, m)
    ry = np.linspace(-n/2, n/2, n)
    U,V = np.meshgrid(rx, ry)

    #ecuacion de turbulencia
    H= np.exp(-k*(U**2+V**2)**(5/6))
    H= np.fft.fftshift(H)

    #aplicacion del filtro
    FT=H*F 
    ft=np.real(np.fft.ifft2(FT))
    return ft

#programa ppal
img = cv2.imread('cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

k=0.05
output= ruido_turbulento(gray,k)

cv2.imshow('Imagen Original',gray)
cv2.imshow('Ruido Turbulento',output)
cv2.waitKey(0)
