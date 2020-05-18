import cv2
import numpy as np
import scipy.ndimage as ndi
import statistics as sts

def filtro_moda(A):
    try:
        S= A.flatten()
        md=sts.mode(S)
    except:
        md=0

    return md

img = cv2.imread('cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Ruido impulsional
mat_noise=np.random.random(gray.shape); #creates a uniform random variable from 0 to 1 

sp_noise_white= np.uint8(np.where(mat_noise>=0.95, 255,0))
sp_noise_black= np.uint8(np.where(mat_noise>=0.05,  1,0))
noise_img = cv2.multiply(gray,sp_noise_black)
noise_img = cv2.add(noise_img,sp_noise_white)

#normalizamos la imagen a rango de 0 a 255
#noise_img = cv2.normalize(noise_img.astype('float'), None, 0, 255, cv2.NORM_MINMAX)

#aplicamos el filtro promedio alfa-acotado
filtro= ndi.generic_filter(noise_img,filtro_moda, [3,3])

cv2.imshow('Imagen original', noise_img)
cv2.imshow('Fitro Moda', filtro)
cv2.waitKey(0)


