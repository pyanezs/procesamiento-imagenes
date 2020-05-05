import cv2
import numpy as np

img = cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE)

#Creando Ruido
m,n = img.shape
mat_noise=np.random.rand(m,n); #creates a uniform random variable from 0 to 1 
sp_noise_white= np.uint8(np.where(mat_noise>=0.9, 255,0))
sp_noise_black= np.uint8(np.where(mat_noise>=0.1,   1,0))

noise_img = cv2.multiply(img,sp_noise_black)
noise_img = cv2.add(noise_img,sp_noise_white)

#desplegar resultados
cv2.imshow('sp',noise_img)
cv2.waitKey(0)

img_median = cv2.medianBlur(noise_img,3)

cv2.imshow('image',img_median)
cv2.waitKey(0)
