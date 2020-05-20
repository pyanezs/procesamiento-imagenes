import cv2
import numpy as np
import scipy.ndimage as ndi

def filtro_ruido_local(A):

    var_N = 0.0009   #varianza estimada 
    B = A.flatten()
    n = len(B)
    var_L = np.var(B) #varianza en la mascara
    
    mu    = np.mean(B)
    g     = B[np.uint8(n/2)]     
    f     =  g - (var_N/var_L)*(g-mu)
    return f   
        
#programa ppal.
img = cv2.imread('cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_norm = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


#generar ruido aleatorio
noise = np.random.random(gray.shape)*0.1
noise_img = gray_norm + noise

#aplicamos el filtro ruido local
filtro = ndi.generic_filter(noise_img, filtro_ruido_local, [7,7])

#estimar el cambio
dif= filtro-noise_img
delta= np.sum(dif.flatten())
print(delta)

cv2.imshow('Imagen con Ruido', noise_img)
cv2.imshow('Fitro Ruido-local', filtro)


cv2.waitKey(0)




