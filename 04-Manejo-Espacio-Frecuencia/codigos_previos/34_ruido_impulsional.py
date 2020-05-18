import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.ndimage as ndi
def filtro_promedio_podado(A):
    d=3
    S= np.sort(A.flatten())
    B= S[d:-d]
    L= len(B)
    val=(1.0/L)*np.sum(B)
    return val



img = cv2.imread('cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Ruido impulsional
mat_noise=np.random.random(gray.shape); #creates a uniform random variable from 0 to 1 

sp_noise_white= np.uint8(np.where(mat_noise>=0.9, 255,0))
sp_noise_black= np.uint8(np.where(mat_noise>=0.05,  1,0))
noise_img = cv2.multiply(gray,sp_noise_black)
noise_img = cv2.add(noise_img,sp_noise_white)
test= ndi.generic_filter(noise_img,filtro_promedio_podado, [3,3])

cv2.imshow('Ruido impulsional', test)
cv2.waitKey(0)


