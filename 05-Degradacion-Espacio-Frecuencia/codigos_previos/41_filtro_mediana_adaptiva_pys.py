import cv2
import numpy as np
import scipy.ndimage as ndi

def filtro_mediana_adaptiva(img, s_max= 16, ws=4):

    img_out = np.zeros(img.shape)
    m, n = img.shape
    coords = [(i, j) for i in range(m - s_max) for j in range(n - s_max)]

    for coord in coords:
        i, j = coord
        zxy = img[i, j]
        wadapt = ws

        while True:
            roi = img[i:(i + wadapt), j:(j + wadapt)]

            size = wadapt <= s_max
            A = condition_A(roi)
            B = condition_B(roi, zxy)

            if size and A and B:
                img_out[i, j] = zxy
                break
            elif size and A and not B:
                img_out[i, j] = np.median(roi.flatten())
                break
            elif size and not A:
                wadapt += 1
                break
            else:
                img_out[i, j] = zxy
                break

    return (np.uint8(img_out[0:(m - s_max), 0:(n-s_max)]))


def condition_A(img):

    z_min = np.min(img.flatten())
    z_max= np.max(img.flatten())
    z_med= np.median(img.flatten())

    a1= z_med - z_min
    a2= z_med - z_max

    return (a1 > 0) & (a2 < 0)


def condition_B(img, zxy):

    z_min = np.min(img.flatten())
    z_max= np.max(img.flatten())

    b1 = float(zxy)-float(z_min)
    b2 = float(zxy)-float(z_max)

    return (b1>0) & (b2<0)


#programa ppal
img = cv2.imread('Fotos/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Ruido impulsional
mat_noise=np.random.random(gray.shape); #creates a uniform random variable from 0 to 1

sp_noise_white= np.uint8(np.where(mat_noise>=0.9, 255,0))
sp_noise_black= np.uint8(np.where(mat_noise>=0.1,  1,0))
noise_img = cv2.multiply(gray,sp_noise_black)
noise_img = cv2.add(noise_img,sp_noise_white)


#aplicamos el filtro mediana_adaptiva
filtro = filtro_mediana_adaptiva(noise_img)

cv2.imshow('Imagen original', noise_img)
cv2.waitKey(0)
cv2.imshow('Fitro Alfa-acotado', filtro)
cv2.waitKey(0)


