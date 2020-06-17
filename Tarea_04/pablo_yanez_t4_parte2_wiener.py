# Libraries
import os
import cv2
import sys
import numpy as np
from pathlib import Path


def btw(M, N, n, fc):

   # M: filas de la imagen
   # N: columnas de la imagen
   # n: orden del filtro
   # fc: frecuencia de corte
   # Calculo de la malla
    vx = np.linspace(-M/2, M/2, M)
    vy = np.linspace(-N/2, N/2, N)
    U, V = np.meshgrid(vy, vx)
    f = np.sqrt(U**2+V**2)

    #filtro de butterworth centrado
    F = 1/(1 + (f/fc)**(2*n))
    F = np.fft.fftshift(F)

    return(F)

def main(args):
    '''Main'''

    ########################################################################
    # Directorio de trabajo
    wd = os.path.join("outs", "wiener")
    Path(wd).mkdir(parents=True, exist_ok=True)

    ########################################################################
    # Imagen segun instrucciones
    input_file = os.path.join("inputs", "cameraman.png")
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    gray = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    F = np.fft.fft2(gray)

    vector = np.linspace(-0.5, 0.5, gray.shape[0])
    U, V = np.meshgrid(vector, vector)
    a = 5
    b = 1
    UV = U*a+V*b
    G = F*np.fft.fftshift(np.sinc(np.pi*UV)*np.exp(-1j*np.pi*UV))
    g = np.real(np.fft.ifft2(G))

    g = cv2.normalize(g.astype('float'), None, 0.0, 255, cv2.NORM_MINMAX)
    g = np.uint8(g)

    out_file = os.path.join(wd, "img.jpg")
    cv2.imwrite(out_file, g)

    ########################################################################
    # Imagen
    noisy = g
    cv2.imshow("Imagen de entrada", noisy)

    ########################################################################
    # H: Pasabajos
    H = btw(256, 256, 2, 30)


    for index, K in enumerate(np.linspace(1e-6, 1e-1, 30)):
        UV = U*a+V*b

        W = np.conj(H)/(np.abs(H)**2 + K)
        G = np.fft.fft2(noisy)
        F = W*G
        filtered = np.real(np.fft.ifft2(F))

        filtered = cv2.normalize(
            filtered.astype('float'),
            None,
            0.0,
            255,
            cv2.NORM_MINMAX)

        out_file = os.path.join(wd, f"filtered__{index}.jpg")
        cv2.imwrite(out_file, filtered)

    for index, K in enumerate(np.linspace(1e-6, 1e-1, 30)):
        print(f"{index} - {K}")


    # img_fft = np.fft.fft2(g)



    # H


    # for K in np.arange(0.000001, 0.001, 0.0001):

    # W = np.conj(H)/(np.abs(H)**2 + K)
    # F = W*G
    # iRestored = np.real(np.fft.ifft2(F))
    # cv2.waitKey(0)


if __name__ == '__main__':
    exit(main(sys.argv))
