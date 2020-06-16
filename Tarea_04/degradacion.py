# Libraries
import os
import cv2
import sys
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, draw, pause


#######################################################################
# Directorio de trabajo
wd = os.path.join("outs", "degradacion")
Path(wd).mkdir(parents=True, exist_ok=True)

#######################################################################
# img
INPUT_FILE = os.path.join("outs", "gaussiano", "img.jpg")

img = cv2.imread(INPUT_FILE, cv2.IMREAD_GRAYSCALE)
img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

print(img.shape)

cv2.imshow("img", img)

# Espectro imagen
img_fft = np.fft.fft2(img)
n, m = img_fft.shape

#######################################################################
######################### Modelo Turbulencia ##########################
for i in [1, 5, 9]:
    k = i / 1000
    rx = np.linspace(-m/2, m/2, m)
    ry = np.linspace(-n/2, n/2, n)
    U, V = np.meshgrid(rx, ry)

    # ecuacion de turbulencia
    H = np.exp(-k*(U**2+V**2)**(5/6))
    H = np.fft.fftshift(H)

    # Aplicacion del filtro
    FT = H * img_fft
    ft = np.real(np.fft.ifft2(FT))

    ft = cv2.normalize(ft.astype('float'), None, 0.0, 255, cv2.NORM_MINMAX)
    ft = np.uint8(ft)

    out_file = os.path.join(wd, f"turb_{str(i).zfill(2)}.jpg")
    cv2.imwrite(out_file, ft)

#######################################################################
########################## Modelo Movimiento ##########################
# Parametros modelo de movimiento
av = [0, 3]
bv = [0, 3]
Tv = [1]

params = [(a, b, T) for a in av for b in bv for T in Tv]

rx = np.linspace(-1, 1, m)
ry = np.linspace(-1, 1, n)
U, V = np.meshgrid(rx, rx)

for a, b, T in params:
    # Ecuacion de movimiento
    UV = (U * a) + (V * b)

    H = T * np.sinc(np.pi*UV)*np.exp(-1j*np.pi*UV)
    H = np.fft.fftshift(H)

    # Aplicacion del filtro
    FT = H * img_fft
    ft = np.real(np.fft.ifft2(FT))

    ft = cv2.normalize(ft.astype('float'), None, 0.0, 255, cv2.NORM_MINMAX)
    ft = np.uint8(ft)

    f_name = f"mov_{a}_{b}_{T}.jpg"
    out_file = os.path.join(wd, f_name)
    cv2.imwrite(out_file, ft)



























#######################################################################
cv2.waitKey(0)
