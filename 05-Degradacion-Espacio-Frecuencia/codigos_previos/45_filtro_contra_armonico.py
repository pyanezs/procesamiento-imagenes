import cv2
import numpy as np
import scipy.ndimage as ndi

def filtro_contra_armonico(A, Q):
    S= np.sum(A.flatten()**(Q+1)) / np.sum(A.flatten()**Q)
    return S


img = cv2.imread('cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Ruido impulsional
mat_noise=np.random.random(gray.shape); #creates a uniform random variable from 0 to 1
sp_noise_black= np.uint8(np.where(mat_noise>=0.2, 1,0))
noise_img = cv2.multiply(gray,sp_noise_black)

#aplicamos el filtro promedio alfa-acotado
filtro= ndi.generic_filter(noise_img,filtro_contra_armonico, [3,3], extra_keywords={'Q':20})

cv2.imshow('Imagen original', noise_img)
cv2.waitKey(0)
cv2.imshow('Fitro Geometrico', filtro)
cv2.waitKey(0)


