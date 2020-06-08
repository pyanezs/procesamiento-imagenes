import cv2
import numpy as np
import scipy.ndimage as ndi

def filtro_geometrico(A):
    largo=  len(A)
    S= np.prod(A.flatten())**(1/largo)
    return S


img = cv2.imread('Fotos/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Ruido impulsional
mat_noise=np.random.random(gray.shape); #creates a uniform random variable from 0 to 1
sp_noise_white= np.uint8(np.where(mat_noise>=0.9, 255,0))

noise_img = cv2.add(gray,sp_noise_white)

#aplicamos el filtro promedio alfa-acotado
filtro= ndi.generic_filter(noise_img,filtro_geometrico, [7,7])

cv2.imshow('Imagen original', noise_img)
cv2.imshow('Fitro Geometrico', filtro)
cv2.waitKey(0)


